# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, DepthwiseSeparableConvModule)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

import math
import sys
# sys.path.append('/home/changkang.li/mmclassification/mmcls/models/backbones')
# sys.path.append('/home/changkang.li/mmclassification/mmcls/models')
from base_backbone import BaseBackbone
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS



# from ..builder import BACKBONES
# from .base_backbone import BaseBackbone

eps = 1.0e-5

class GhostModule(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GhostModule, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        self.ratio = ratio
        self.dw_size = dw_size
        self.dw_dilation = (dw_size - 1) // 2
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.init_channels, kernel_size, self.stride, padding=self.padding)

        # self.conv1 = DepthwiseSeparableConvModule(self.in_channels, self.init_channels, kernel_size = kernel_size, stride=self.stride, padding=self.padding, norm_cfg=None, act_cfg=None)
        self.conv2 = nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, padding=int(self.dw_size/2), groups=self.init_channels) # 32 x 32
        
        
        # self.weight1 = nn.Parameter(torch.Tensor(self.init_channels, self.in_channels, kernel_size, kernel_size))
        self.weight1 = nn.Parameter(torch.Tensor(self.init_channels, 1, kernel_size, kernel_size))
        self.bn1 = nn.BatchNorm2d(self.init_channels)
        if self.new_channels > 0:
            self.weight2 = nn.Parameter(torch.Tensor(self.new_channels, 1, self.dw_size, self.dw_size))
            self.bn2 = nn.BatchNorm2d(self.out_channels - self.init_channels)
        
        if bias:
            self.bias =nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.new_channels > 0:
            nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input): 
        x1 = self.conv1(input) # 1 32 56 56
        if self.new_channels == 0:
            return x1
        x2 = self.conv2(x1)  # 1 32 56 56
        x2 = x2[:, :self.out_channels - self.init_channels, :, :] # 1 32 
        x = torch.cat([x1, x2], 1)
        return x

# 改basicblock ---> ResNetBasicBlock
class BasicBlock(BaseModule):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=2,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None,
                 scale_yasuo=2):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        # assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.scale_yasuo = scale_yasuo

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        # self.conv1 = build_conv_layer(
        #     conv_cfg,
        #     in_channels,
        #     self.mid_channels,
        #     3,
        #     stride=stride,
        #     padding=dilation,
        #     dilation=dilation,
        #     bias=False)
        self.conv1 = DepthwiseSeparableConvModule(
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = DepthwiseSeparableConvModule(
            # conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

# 改basicblock ---> BasicBlock_slim
class BasicBlock_slim(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None,
                 scale_factor=3):
        super(BasicBlock_slim, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion // scale_factor
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        # self.conv1 = build_conv_layer(
        #     conv_cfg,
        #     in_channels,
        #     self.mid_channels,
        #     3,
        #     stride=stride,
        #     padding=dilation,
        #     dilation=dilation,
        #     bias=False)
        self.conv1 = DepthwiseSeparableConvModule(
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = DepthwiseSeparableConvModule(
            # conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# 改basicblock ---> BasicBlock_ghost
class BasicBlock_ghost(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None,
                 scale_factor=1,
                 ratio=2):
        super(BasicBlock_ghost, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion // scale_factor
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.ratio = ratio

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = GhostModule(
            in_channels,
            self.mid_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            ratio=ratio)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = GhostModule(
            # conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            ratio=ratio)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out



class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class Res_depNet(BaseBackbone):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.

    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
        # 50: (Bottleneck, (3, 4, 6, 3)),
        # 101: (Bottleneck, (3, 4, 23, 3)),
        # 152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                #  expansion=2,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 deep_stem_cp=False,
                 dw_kse = False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0,
                 scale_factor=1,
                 slim=False,
                 ghost=False,
                 ratio=2):
        super(Res_depNet, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for Res_depNet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.deep_stem_cp = deep_stem_cp
        self.dw_kse = dw_kse
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)
        self.scale_factor = scale_factor
        self.slim=slim
        self.ghost=ghost
        self.ratio=ratio

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        
        if self.ghost==True:
            self.res_layers = []
            _in_channels = stem_channels
            _out_channels = base_channels * self.expansion
            for i, num_blocks in enumerate(self.stage_blocks):
                stride = strides[i]
                dilation = dilations[i]
                res_layer = self.make_res_layer(
                    block=BasicBlock_ghost,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate,
                    scale_factor=self.scale_factor,
                    ratio=self.ratio)
                _in_channels = _out_channels
                _out_channels *= 2
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

            self._freeze_stages()

            self.feat_dim = res_layer[-1].out_channels
        

        elif not self.slim:
            self.res_layers = []
            _in_channels = stem_channels
            _out_channels = base_channels * self.expansion
            for i, num_blocks in enumerate(self.stage_blocks):
                stride = strides[i]
                dilation = dilations[i]
                res_layer = self.make_res_layer(
                    block=self.block,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate)
                _in_channels = _out_channels
                _out_channels *= 2
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

            self._freeze_stages()

            self.feat_dim = res_layer[-1].out_channels
        # self.slim = True
        else:
            self.res_layers = []
            _in_channels = stem_channels
            _out_channels = base_channels * self.expansion
            for i, num_blocks in enumerate(self.stage_blocks):
                stride = strides[i]
                dilation = dilations[i]
                res_layer = self.make_res_layer(
                    block=BasicBlock_slim,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate,
                    scale_factor=self.scale_factor)
                _in_channels = _out_channels
                _out_channels *= 2
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

            self._freeze_stages()

            self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem_cp:
            self.stem_cp = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                DepthwiseSeparableConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                DepthwiseSeparableConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        elif self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem_cp:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            elif self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(Res_depNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem_cp:
            x = self.stem_cp(x)
        elif self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(Res_depNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class Res_depNetV1c(Res_depNet):
    """ResNetV1c backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.
    """

    def __init__(self, **kwargs):
        super(Res_depNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class Res_depNetV1d(Res_depNet):
    """ResNetV1d backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(Res_depNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)



@BACKBONES.register_module()
class Res_depNetV1d_cp(Res_depNet):
    """ResNetV1d backbone.

    This variant is described in `Bag of Tricks.
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(Res_depNetV1d_cp, self).__init__(
            deep_stem_cp=True, avg_down=True, **kwargs)


@BACKBONES.register_module()
class Res_depNetV1d_cp_slim(Res_depNet):
    # 压缩三倍
    def __init__(self, **kwargs):
        super(Res_depNetV1d_cp_slim, self).__init__(
            slim=True, deep_stem_cp=True, avg_down=True, **kwargs)


@BACKBONES.register_module()
class Res_depNetV1d_cp_ghost(Res_depNet):
    # ghost块
    def __init__(self, **kwargs):
        super(Res_depNetV1d_cp_ghost, self).__init__(
            ghost=True, deep_stem_cp=True, avg_down=True, **kwargs)



class Res_depNetV1d_cp_kse(Res_depNetV1d_cp):

    def __init__(self, **kwargs):
        super(Res_depNetV1d_cp_kse, self).__init__(
            dw_kse = True, deep_stem_cp=True, avg_down=True, **kwargs)

    def KSE(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T
        weight = self.weight.data.cpu().numpy()

        # Calculate channels KSE indicator.
        weight = weight.transpose(1, 0, 2, 3).reshape(self.input_channels, self.output_channels, -1)  # C_out, C_in, K*K
        ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)   # ks_weight = sum(|w|_1)
        ke_weight = density_entropy(weight.reshape(self.output_channels, self.input_channels, -1)) # ke_weight = entropy()
        ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))  # normalize
        ke_weight = (ke_weight - np.min(ke_weight)) / (np.max(ke_weight) - np.min(ke_weight))  # normalize
        indicator = np.sqrt(ks_weight / (1 + ke_weight))    # 
        indicator = (indicator - np.min(indicator))/(np.max(indicator) - np.min(indicator)) # normalize

        # Calculate each input channels kernel number and split them into different groups.
        # Each group has same kernel number.
        mask = np.zeros(shape=(self.input_channels))  # 记录每个in channel属于哪个group
        self.group_size = [0 for i in range(self.G)] # G=4 每个group的大小
        self.cluster_num = [1 for i in range(self.G)] # qc决定 每个group的 kernel数

        for i in range(self.input_channels):  # 计算groupsize
            if math.floor(indicator[i] * self.G) == 0: # 下取整，
                mask[i] = 0
                self.group_size[0] += 1
            elif math.ceil(indicator[i] * self.G) == self.G:
                mask[i] = self.G - 1
                self.group_size[-1] += 1
            else:
                mask[i] = math.floor(indicator[i] * self.G)  # 分类取决于 KSE indicator vc（[0,1]）的值。
                self.group_size[int(math.floor(indicator[i] * self.G))] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G)) # 文中的qc 需要为该输入通道c 输出聚类数 保留几个kernel？
        self.mask.data = torch.Tensor(mask)

        # Generate corresponding cluster and index.
        # For kernel number = N: use full_weight rather than cluster&index
        self.full_weight = nn.Parameter(torch.Tensor(
            self.output_channels, self.group_size[-1], self.kernel_size, self.kernel_size),requires_grad=True)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            else:
                cluster = nn.Parameter(torch.Tensor(
                    self.cluster_num[g], self.group_size[g], self.kernel_size, self.kernel_size), requires_grad=True)
                index = nn.Parameter(torch.zeros(self.output_channels, self.group_size[g]), requires_grad=False)
                self.__setattr__("clusters_" + str(g), cluster)
                self.__setattr__("indexs_" + str(g), index)

        # Calculate cluster and index by k-means
        # First, collect weight corresponding to each group(same kernel number)
        weight = self.weight.data.cpu().numpy()
        weight_group = []  # 分组后的weight （输入维度c上）
        for g in range(self.G):  # weight分组 到 weight_group
            if self.group_size[g] == 0:
                weight_group.append([])
                continue
            each_weight_group = []
            for c in range(self.input_channels):
                if mask[c] == g:  # 分类
                    each_weight_group.append(np.expand_dims(weight[:, c], 1))
            each_weight_group = np.concatenate(each_weight_group, 1)
            weight_group.append(each_weight_group)

        self.full_weight.data = torch.Tensor(weight_group[-1])

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster_weight = weight_group[g]
            cluster_weight = cluster_weight.transpose((1, 0, 2, 3)).reshape((
                self.group_size[g], self.output_channels, -1))  # Cin_group * Cout * K*K

            clusters = []  # 
            indexs = []
            for c in range(cluster_weight.shape[0]):  # 在压缩后的每个Cin通道上进行聚类
                kmean = KMeans(n_clusters=self.cluster_num[g]).fit(cluster_weight[c]) # 对每个C_cin_group(每个group)进行聚类
                centroids = kmean.cluster_centers_ # 聚类中心
                assignments = kmean.labels_

                clusters.append(np.expand_dims(np.reshape(centroids, [
                    self.cluster_num[g], self.kernel_size, self.kernel_size]), 1)) # 最终得到 C_group * N_out * K*K
                indexs.append(np.expand_dims(assignments, 1)) # 存放

            clusters = np.concatenate(clusters, 1) # 每一类的weight，（8,5,3,3）
            indexs = np.concatenate(indexs, 1)  # 

            self.__getattr__("clusters_"+str(g)).data = torch.Tensor(clusters)
            self.__getattr__("indexs_"+str(g)).data = torch.Tensor(indexs)

    def forward_init(self):
        # record the channel index of each group

        full_index = []     # input channels index which kernel number = N
        cluster_indexs = [] # input channels index which kernel number != N && != 0 最终返回4个
        all_indexs = []     # input channels index which kernel number != 0

        for i, m in enumerate(self.mask.data):  # mask记录每个channel所在的G i：Cin m:G_index
            if m == self.G - 1:
                full_index.append(i)
                all_indexs.append(i)

        for g in range(1, self.G - 1):  # 每group,就是分类储存在不同的列表里，G=0 的三个直接不要了
            if self.group_size[g] == 0:
                cluster_indexs.append([])
                continue
            cluster_index = []
            for i, m in enumerate(self.mask.data):
                if m == g:  # Cin 通道，分类到 cluster_index
                    cluster_index.append(i)
                    all_indexs.append(i)
            cluster_indexs.append(cluster_index)

        self.channels_indexs = nn.Parameter(torch.zeros(self.input_channels - self.group_size[0]).long(),
                                            requires_grad=False)

        # transform index for training
        if self.full_weight.is_cuda:
            self.channels_indexs.data = torch.LongTensor(all_indexs).cuda()
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1).cuda()).view(-1).long(),
                    requires_grad=False))

        else:
            self.channels_indexs.data = torch.LongTensor(all_indexs)
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1)).view(-1).long(),
                    requires_grad=False))

if __name__ == '__main__':
    import torch
    from torch.nn import functional as F
    # x = torch.randn(1, 3, 28, 28) 
    # w = torch.rand(16, 3, 5, 5)  # 16种3通道的5乘5卷积核
    # b = torch.rand(16) 
    # out = F.conv2d(x, w, b, stride=1, padding=1) 
    # model = Res_depNetV1d_cp_slim(depth=18, scale_factor=3)
    model = Res_depNetV1d_cp_ghost(depth=18, scale_factor=2, ratio=4)
    model.init_weights()
    # 显示网络结构
    print(model)
    x = torch.randn(1, 3, 224, 224)
    outs = model(x)
    print([out.shape for out in outs])

    from mmcv.cnn.utils import get_model_complexity_info
    model.eval()

    flops, params = get_model_complexity_info(model, (3, 224, 224))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {(3, 224, 224)}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

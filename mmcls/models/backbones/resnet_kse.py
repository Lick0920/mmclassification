# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
import math
import numpy as np
import time
import pdb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from math import isnan
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ###
# import sys
# sys.path.append('/home/changkang.li/Projects/mmclassification/mmcls/models/backbones')
# from base_backbone import BaseBackbone
# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.utils import Registry
# MODELS = Registry('models', parent=MMCV_MODELS)
# BACKBONES = MODELS
###
eps = 1.0e-5


class Conv2d_KSE(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=True, G=4, T=0):
        super(Conv2d_KSE, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.isbias = bias
        self.T = T

        if G == 0:
            if output_channels >= input_channels:
                self.G = input_channels
            else:
                self.G = math.ceil(input_channels/output_channels)
        else:
            self.G = G
        self.group_num = self.G
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.Tensor(input_channels), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def __repr__(self):
        return self.__class__.__name__ \
               + "({" + str(self.input_channels) \
               + "}, {" + str(self.output_channels) \
               + "}, kernel_size={" + str(self.kernel_size) + "}, stride={" + \
               str(self.stride) + "}, padding={" + str(self.padding) + "})"

    def forward(self, input):
        # transform cluster and index into weight for training
        cluster_weights = []
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            clusters = cluster.permute(1, 0, 2, 3).contiguous().view(
                self.cluster_num[g] * cluster.shape[1], self.kernel_size, self.kernel_size)
            cluster_weight = clusters[
                self.__getattr__("cluster_indexs_" + str(g)).data].contiguous().view(
                self.output_channels, cluster.shape[1], self.kernel_size, self.kernel_size)

            cluster_weights.append(cluster_weight)

        if len(cluster_weights) == 0:
            weight = self.full_weight
        else:
            weight = torch.cat((self.full_weight, torch.cat(cluster_weights, dim=1)), dim=1)

        select_indexs = torch.index_select(input, 1, self.channels_indexs)  # 1 12 32 32
        return F.conv2d(torch.index_select(input, 1, self.channels_indexs), weight, self.bias,
                        stride=self.stride,
                        padding=self.padding)

    
    def density_entropy(self, X): # 计算密度熵， X: N*C*D
        K = 5  # 经验值
        N, C, D = X.shape
        x = X.transpose(1, 0, 2).reshape(C, N, -1) # C*N*D
        score = []
        for c in range(C):
            # print(x[c])
            nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c]) # K近邻
            dms = []
            for i in range(N):
                dm = 0
                dist, ind = nbrs.kneighbors(x[c, i].reshape(1, -1))
                for j, id in enumerate(ind[0][1:]):
                    dm += dist[0][j + 1]

                dms.append(dm)

            dms_sum = sum(dms)
            en = 0
            for i in range(N):
                en += -dms[i]/dms_sum*math.log(dms[i]/dms_sum, 2) # 计算熵

            score.append(en)
        return np.array(score)

    def KSE(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T
        weight = self.weight.data.cpu().numpy()

        # Calculate channels KSE indicator.
        weight = weight.transpose(1, 0, 2, 3).reshape(self.input_channels, self.output_channels, -1)  # C_out, C_in, K*K
        ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)   # ks_weight = sum(|w|_1)
        ke_weight = self.density_entropy(weight.reshape(self.output_channels, self.input_channels, -1)) # ke_weight = entropy()
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

    def create_arch(self, G=None, T=None):  # test 时有用
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T

        # create architecture (clusters and indexs) base on mask
        mask = self.mask.data.cpu().numpy()
        self.group_size = [0 for i in range(self.G)]
        self.cluster_num = [1 for i in range(self.G)]

        for i in range(self.input_channels):
            self.group_size[int(mask[i])] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G))

        self.full_weight = nn.Parameter(
            torch.Tensor(self.output_channels, self.group_size[-1], self.kernel_size, self.kernel_size),
            requires_grad=True)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            else:
                cluster = nn.Parameter(torch.zeros(
                    self.cluster_num[g], self.group_size[g], self.kernel_size, self.kernel_size), requires_grad=True)
                index = nn.Parameter(torch.ByteTensor(math.ceil(
                    math.ceil(math.log(self.cluster_num[g], 2)) * self.output_channels * self.group_size[g] / 8)),
                    requires_grad=False)
                self.__setattr__("clusters_" + str(g), cluster)
                self.__setattr__("indexs_" + str(g), index)

    def load(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]
            bits = math.ceil(math.log(Q, 2))
            indexs = index.data.cpu().numpy()
            new_b = ""
            f = '{0:08b}'

            for n, i in enumerate(indexs):
                if (self.output_channels*self.group_size[g]*bits)% 8 != 0 and n == indexs.shape[0]-1:
                    continue
                new_b += f.format(i)

            if (self.output_channels * self.group_size[g] * bits) % 8 != 0:
                va = (self.output_channels * self.group_size[g] * bits) % 8
                new_b += f.format(indexs[-1])[-va:]

            new_index = []

            for i in range(int(len(new_b)/bits)):
                b = new_b[i*bits:(i+1)*bits]
                v = float(int(b, 2))
                new_index.append(v)

            ni = torch.Tensor(new_index).view(self.output_channels, self.group_size[g])
            self.__delattr__("indexs_" + str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(ni, requires_grad=False))

    def save(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]

            bits = math.ceil(math.log(Q, 2))

            new_b = ""
            indexs = index.data.cpu().numpy().reshape(-1)
            f = '{0:0'+str(bits)+'b}'
            for i in indexs:
                nb = f.format(int(i))
                new_b += nb

            new_index = []
            for i in range(int(len(new_b)*1.0/8)):
                new_value = int(new_b[i*8:(i+1)*8], 2)
                new_index.append(new_value)

            if len(new_b) % 8 != 0:
                new_value = int(new_b[int(len(new_b)*1.0/8)*8:], 2)
                new_index.append(new_value)

            self.__delattr__("indexs_"+str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(torch.ByteTensor(new_index), requires_grad=False))
            self.__delattr__("cluster_indexs_" + str(g))
        self.__delattr__("channels_indexs")
        self.__delattr__("group_size")
        self.__delattr__("cluster_num")
        self.__delattr__("weight")
        if self.bias is not None:
            self.__delattr__("bias")


# class BasicBlock(BaseModule):
#     """BasicBlock for ResNet.

#     Args:
#         in_channels (int): Input channels of this block.
#         out_channels (int): Output channels of this block.
#         expansion (int): The ratio of ``out_channels/mid_channels`` where
#             ``mid_channels`` is the output channels of conv1. This is a
#             reserved argument in BasicBlock and should always be 1. Default: 1.
#         stride (int): stride of the block. Default: 1
#         dilation (int): dilation of convolution. Default: 1
#         downsample (nn.Module, optional): downsample operation on identity
#             branch. Default: None.
#         style (str): `pytorch` or `caffe`. It is unused and reserved for
#             unified API with Bottleneck.
#         with_cp (bool): Use checkpoint or not. Using checkpoint will save some
#             memory while slowing down the training speed.
#         conv_cfg (dict, optional): dictionary to construct and config conv
#             layer. Default: None
#         norm_cfg (dict): dictionary to construct and config norm layer.
#             Default: dict(type='BN')
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  expansion=1,
#                  stride=1,
#                  dilation=1,
#                  downsample=None,
#                  style='pytorch',
#                  with_cp=False,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN'),
#                  drop_path_rate=0.0,
#                  act_cfg=dict(type='ReLU', inplace=True),
#                  init_cfg=None):
#         super(BasicBlock, self).__init__(init_cfg=init_cfg)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.expansion = expansion
#         assert self.expansion == 1
#         assert out_channels % expansion == 0
#         self.mid_channels = out_channels // expansion
#         self.stride = stride
#         self.dilation = dilation
#         self.style = style
#         self.with_cp = with_cp
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg

#         self.norm1_name, norm1 = build_norm_layer(
#             norm_cfg, self.mid_channels, postfix=1)
#         self.norm2_name, norm2 = build_norm_layer(
#             norm_cfg, out_channels, postfix=2)

#         self.conv1 = build_conv_layer(
#             conv_cfg,
#             in_channels,
#             self.mid_channels,
#             3,
#             stride=stride,
#             padding=dilation,
#             dilation=dilation,
#             bias=False)
#         self.add_module(self.norm1_name, norm1)
#         self.conv2 = build_conv_layer(
#             conv_cfg,
#             self.mid_channels,
#             out_channels,
#             3,
#             padding=1,
#             bias=False)
#         self.add_module(self.norm2_name, norm2)

#         self.relu = build_activation_layer(act_cfg)
#         self.downsample = downsample
#         self.drop_path = DropPath(drop_prob=drop_path_rate
#                                   ) if drop_path_rate > eps else nn.Identity()

#     @property
#     def norm1(self):
#         return getattr(self, self.norm1_name)

#     @property
#     def norm2(self):
#         return getattr(self, self.norm2_name)

#     def forward(self, x):

#         def _inner_forward(x):
#             identity = x

#             out = self.conv1(x)
#             out = self.norm1(out)
#             out = self.relu(out)

#             out = self.conv2(out)
#             out = self.norm2(out)

#             if self.downsample is not None:
#                 identity = self.downsample(x)

#             out = self.drop_path(out)

#             out += identity

#             return out

#         if self.with_cp and x.requires_grad:
#             out = cp.checkpoint(_inner_forward, x)
#         else:
#             out = _inner_forward(x)

#         out = self.relu(out)

#         return out

# 加了conv_kse
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
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
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
        self.conv1 = Conv2d_KSE(
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            bias=False)
        # weight init
        init.kaiming_normal(self.conv1.weight)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = Conv2d_KSE(
            # conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        init.kaiming_normal(self.conv2.weight)

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
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
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
class ResNet_kse(BaseBackbone):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
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
                 drop_path_rate=0.0):
        super(ResNet_kse, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
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

        self._make_stem_layer(in_channels, stem_channels)

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

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
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
            if self.deep_stem:
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
        super(ResNet_kse, self).init_weights()

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
        if self.deep_stem:
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
        super(ResNet_kse, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()




if __name__ == '__main__':
    model = ResNet_kse(depth=18)
    print(model)
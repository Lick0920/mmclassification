model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Res_depNetV1d',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude', magnitude_range=(0,
                                                                       0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=160),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(
                type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
            dict(
                type='Posterize', magnitude_key='bits', magnitude_range=(4,
                                                                         0)),
            dict(
                type='Solarize', magnitude_key='thr', magnitude_range=(256,
                                                                       0)),
            dict(
                type='SolarizeAdd',
                magnitude_key='magnitude',
                magnitude_range=(0, 110)),
            dict(
                type='ColorTransform',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='horizontal'),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='vertical'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='horizontal'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=6,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(236, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='ImageNet',
        data_prefix='/home/changkang.li/dataset_lck/Imagenet12/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=160),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 30)),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(4, 0)),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(256, 0)),
                    dict(
                        type='SolarizeAdd',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 110)),
                    dict(
                        type='ColorTransform',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=6,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='/home/changkang.li/dataset_lck/Imagenet12/val',
        ann_file=
        '/home/changkang.li/dataset_lck/Imagenet12/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(236, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='/home/changkang.li/dataset_lck/Imagenet12/val',
        ann_file=
        '/home/changkang.li/dataset_lck/Imagenet12/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(236, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=5, metric='accuracy')
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'acheckpoint/res_depnetv1d18_lr_in1k_bs256a3_gpu8'
gpu_ids = range(0, 8)

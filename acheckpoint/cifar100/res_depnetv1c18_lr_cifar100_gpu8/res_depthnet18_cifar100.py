model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Res_depNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.07, 112.434], std=[68.17, 65.392, 70.418], to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR100',
        data_prefix='home/chagnkang.li/dataset_lck/cifar-100-python',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR100',
        data_prefix='home/chagnkang.li/dataset_lck/cifar-100-python',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR100',
        data_prefix='home/chagnkang.li/dataset_lck/cifar-100-python',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
evaluation = dict(interval=10, metric='accuracy')
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'acheckpoint/res_depnetv1c18_lr_cifar100_gpu8'
gpu_ids = range(0, 8)

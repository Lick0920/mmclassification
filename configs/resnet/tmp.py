_base_ = [
    '../_base_/models/resdepnet18_in1k.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_coslr.py', '../_base_/default_runtime.py'
]
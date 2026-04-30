# CenterPoint config — Boston baseline training.
# Inherits from mmdet3d's standard nuScenes config.
# TODO: paste full _base_ inheritance + override data_root, ann_file, max_epochs.

_base_ = [
    # 'mmdet3d::centerpoint/centerpoint_voxel0075_secfpn_dcn_8xb4-cyclic-20e_nus-3d.py',
]

data_root = "/kaggle/input/nuscenes-mini/v1.0-mini/"

train_cfg = dict(max_epochs=5, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2),
    logger=dict(type="LoggerHook", interval=10),
)

# TODO: filter dataset by Boston scene tokens from splits.json

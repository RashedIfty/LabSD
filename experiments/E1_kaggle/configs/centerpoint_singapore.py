# CenterPoint config — Singapore fine-tune (the C1 retrain step in E1).

_base_ = [
    # 'mmdet3d::centerpoint/centerpoint_voxel0075_secfpn_dcn_8xb4-cyclic-20e_nus-3d.py',
]

data_root = "/kaggle/input/nuscenes-mini/v1.0-mini/"

# Resume from Boston checkpoint
load_from = "/kaggle/input/labsd-checkpoints/c1_boston_final.pth"

train_cfg = dict(max_epochs=3, val_interval=1)

# Lower LR for fine-tune
optim_wrapper = dict(optimizer=dict(lr=1e-4))

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2),
    logger=dict(type="LoggerHook", interval=10),
)

# TODO: filter dataset by Singapore scene tokens from splits.json

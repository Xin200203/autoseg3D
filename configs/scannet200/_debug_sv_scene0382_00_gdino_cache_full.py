_base_ = ["./AutoSeg3D_sv_scannet200_A3_fpn_2dca_autoloss.py"]

# One-scene smoke config for verifying GDINO disk cache + 2D-3D alignment.
#
# Usage:
#   export GDINO_CACHE_DIR=/home/nebula/xxy/dataset/gdino/scannet200_sv_scene0382_00_full
#   python tools/train.py configs/scannet200/_debug_sv_scene0382_00_gdino_cache_full.py

scene_id = "scene0382_00"

# Keep paths explicit to avoid cwd-dependent relative paths.
data_root = "/home/nebula/xxy/dataset/data/scannet200-sv/"
train_ann = f"subsets/gdino_{scene_id}_train.pkl"
val_ann = f"subsets/gdino_{scene_id}_val.pkl"

# Use the same cache dir for point_fusion + DACA-2D.
gdino_cache_dir = f"/home/nebula/xxy/dataset/gdino/scannet200_sv_{scene_id}_full"

model = dict(
    gdino_point_fusion=dict(cache_dir=gdino_cache_dir),
    train_cfg=dict(gdino_daca2d=dict(cache_dir=gdino_cache_dir)),
    test_cfg=dict(gdino_daca2d=dict(cache_dir=gdino_cache_dir)),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(data_root=data_root, ann_file=train_ann),
)
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(data_root=data_root, ann_file=val_ann))
test_dataloader = val_dataloader

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(type="CheckpointHook", interval=999, save_last=False, save_best=None),
)

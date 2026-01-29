_base_ = ["./AutoSeg3D_scannet200_stage1_A3_fpn_2dca_autoloss_cat.py"]

# One-scene smoke config for verifying GDINO disk cache + 2D-3D alignment (MV).

scene_id = "scene0382_00"

data_root = "/home/nebula/xxy/dataset/data/scannet200-mv_fast/"
train_ann = f"subsets/gdino_{scene_id}_train.pkl"
val_ann = f"subsets/gdino_{scene_id}_val.pkl"

gdino_cache_dir = f"/home/nebula/xxy/dataset/gdino/scannet200_mv_{scene_id}_full"

model = dict(
    gdino_point_fusion=dict(cache_dir=gdino_cache_dir),
    train_cfg=dict(gdino_daca2d=dict(cache_dir=gdino_cache_dir)),
    # Keep test_cfg unchanged; DACA-2D is used via decoder cfg in this codebase.
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
default_hooks = dict(logger=dict(interval=1))

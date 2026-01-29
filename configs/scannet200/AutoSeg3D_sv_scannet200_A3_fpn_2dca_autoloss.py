_base_ = ['./AutoSeg3D_sv_scannet200_A1_2dca.py']

# A3: ESAM-style sparse-FPN injection (point-wise GDINO feats -> build_sparse_fpn -> UNet decoder fusion) + 2DCA.
#
# Key property: UNet `in_channels` stays 3 (no inflate). 2D features are injected at decoder levels.

model = dict(
    backbone=dict(
        # Enable sparse-FPN injection in MinkUNet decoder (expects [s1,s2,s4,s8,s16] with C=256).
        dino_dim=256,
        config=dict(
            # Multi-scale projection alignment must be strict for reliable ablation.
            dino_strict=True,
            dino_min_hit_ratio=0.95,
            dino_residual=True,
        ),
    ),
    gdino_point_fusion=dict(
        enable=True,
        in_dim=256,
        out_dim=256,
        proj_type='identity',
        # Use sparse-FPN injection (NOT early-fusion).
        mode='fpn',
        fuse_mode='fpn',
        feat_levels=[0, 1, 2, 3],
        backbone_only=True,
        max_depth=10.0,
        align_corners=False,
        strict=True,
        strict_valid_ratio=0.95,
        log_fail=True,
    ),
)

# Keep training schedule identical to SV baseline except `val_interval`.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=8)

# Checkpoint: only keep latest + best AP50 (matches baseline intent; avoids per-epoch clutter).
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='all_ap_50%',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

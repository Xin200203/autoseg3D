_base_ = ['./AutoSeg3D_sv_scannet200_A1_2dca.py']

# A2: point-level early fusion (single GDINO feature level) + 2DCA (DACA-2D).
# Note: UNet in_channels changes from 3 -> 3+256. Use inflated Mask3D checkpoint.

load_from = 'work_dirs/tmp/mask3d_scannet200_in259.pth'

model = dict(
    backbone=dict(in_channels=3 + 256),
    gdino_point_fusion=dict(
        enable=True,
        in_dim=256,
        out_dim=256,
        proj_type='identity',
        feat_levels=[0],
        backbone_only=True,
        max_depth=10.0,
        align_corners=False,
        strict=False,
        strict_valid_ratio=0.95,
        log_fail=True,
    ),
)


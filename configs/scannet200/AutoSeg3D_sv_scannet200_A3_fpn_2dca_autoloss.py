_base_ = ['./AutoSeg3D_sv_scannet200_A1_2dca.py']

# A3: ESAM-style sparse-FPN injection (point-wise GDINO feats -> build_sparse_fpn -> UNet decoder fusion) + 2DCA.
#
# Key property: UNet `in_channels` stays 3 (no inflate). 2D features are injected at decoder levels.

model = dict(
    backbone=dict(
        # Enable sparse-FPN injection in MinkUNet decoder (expects [s1,s2,s4,s8,s16] with C=256).
        dino_dim=256,
        config=dict(
            # Keep default non-strict by default for ablation; can be tightened once stable.
            dino_strict=False,
            dino_min_hit_ratio=0.05,
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
        strict=False,
        strict_valid_ratio=0.95,
        log_fail=True,
    ),
)

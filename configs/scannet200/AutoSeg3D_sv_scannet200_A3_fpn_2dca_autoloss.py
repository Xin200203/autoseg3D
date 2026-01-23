_base_ = ['./AutoSeg3D_sv_scannet200_A2_earlyfusion_2dca.py']

# A3: FPN-style point fusion (multi-level avg) + 2DCA, keep AutoSeg3D loss.

model = dict(
    gdino_point_fusion=dict(
        feat_levels=[0, 1, 2, 3],
        backbone_only=True,
    ),
)


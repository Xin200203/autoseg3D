_base_ = ["./AutoSeg3D_sv_scannet200_A3_fpn_2dca_autoloss_cat.py"]

# Category-aware GDINO object queries (20-cluster caption).
# Tuned by inference-side sweep on subset20_v2:
# - mask.metric='l1', mask.thr=0.7
# - score_thr=0.05
#
# Prompt source (for reference):
#   configs/scannet200/prompts/scannet200_cluster20_gdino.py
GDINO_CLUSTER20_CAPTION = (
    "chair. sofa. stool. bench. table. desk. counter. cabinet. shelf. bookshelf. "
    "wardrobe. bed. pillow. blanket. toilet. sink. bathtub. lamp. tv. monitor."
)

model = dict(
    gdino_backbone=dict(caption=GDINO_CLUSTER20_CAPTION),
    train_cfg=dict(
        gdino_daca2d=dict(
            score_thr=0.05,
            mask=dict(metric="l1", thr=0.7),
        )
    ),
    test_cfg=dict(
        gdino_daca2d=dict(
            score_thr=0.05,
            mask=dict(metric="l1", thr=0.7),
        )
    ),
)

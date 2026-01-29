_base_ = ['./AutoSeg3D_sv_scannet200_A3_fpn_2dca_autoloss.py']

# Category-aware (ScanNet200 thing classes) SV training/eval.
# Keep A3 architecture (sparse-FPN GDINO point fusion + 2DCA) unchanged, but
# switch instance classification from class-agnostic "object" to 198 thing classes.

NUM_THING_CLASSES = 198

# ----------------------------------------------------------------------------
# Model: switch instance class heads + criterion to 198 classes
# ----------------------------------------------------------------------------
model = dict(
    num_classes=NUM_THING_CLASSES,
    decoder=dict(num_instance_classes=NUM_THING_CLASSES),
    criterion=dict(
        inst_criterion=dict(num_classes=NUM_THING_CLASSES),
    ),
    # DACA-2D: use L1 distance mask with thr=0.3
    train_cfg=dict(
        gdino_daca2d=dict(mask=dict(metric='l1', thr=0.3))
    ),
    test_cfg=dict(
        gdino_daca2d=dict(mask=dict(metric='l1', thr=0.3))
    ),
)

# ----------------------------------------------------------------------------
# Evaluator: force category-aware AP (do NOT fallback to cat-agnostic when
# predicted labels happen to be all zeros).
# ----------------------------------------------------------------------------
val_evaluator = dict(cat_agnostic=False)
test_evaluator = val_evaluator

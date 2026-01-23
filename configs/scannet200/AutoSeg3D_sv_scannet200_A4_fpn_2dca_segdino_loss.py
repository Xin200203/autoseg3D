_base_ = ['./AutoSeg3D_sv_scannet200_A3_fpn_2dca_autoloss.py']

# A4: sparse-FPN injection + 2DCA, switch to SegDINO3D-style matcher/loss settings.
# - Enable bbox head in decoder (axis-aligned).
# - Add Center/Size L1 costs to matcher.
# - Add center/size L1 loss weights (score loss weight set to 0 since objectness is disabled).
# - Compute GT bboxes on-the-fly from GT masks + point xyz (SV infos do not provide bboxes_3d).

model = dict(
    decoder=dict(bbox_flag=True),
    criterion=dict(
        inst_criterion=dict(
            matcher=dict(
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                    dict(type='CenterL1Cost', weight=0.5),
                    dict(type='SizeL1Cost', weight=0.5),
                ],
                topk=1),
            loss_weight=[0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5],
        ),
    ),
    train_cfg=dict(compute_gt_bboxes_3d=True),
)

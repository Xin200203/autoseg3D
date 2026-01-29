_base_ = ['./AutoSeg3D_sv_scannet200_A3_fpn_2dca_autoloss.py']

# A5: FPN injection (point-wise GDINO srcs -> sparse-FPN -> UNet decoder fusion)
#   + 2DCA (GDINO DACA-2D, metric=L1, thr=0.3)
#   + SegDINO3D-style 3D box-modulated CA (SP-domain)
#   + SegDINO3D-style loss (add center/size costs + L1 losses)
#
# Notes:
# - Keep SV baseline model class (ScanNet200MixFormer3D, offline, no memory).
# - UNet in_channels stays 3 (no inflate). FPN features are injected at decoder levels.

num_instance_classes = 1
num_semantic_classes = 200

# Paper DACA-2D: L1 distance + thr
_daca_metric = 'l1'
_daca_thr = 0.30

# SegDINO3D CA-3D positional encoding temperature (paper uses T=20)
_box3d_pe_temperature = 20.0

model = dict(
    decoder=dict(
        # Align closer to SegDINO3D depth for the 3D decoder side.
        num_layers=6,
        cross_attn_mode=["", "SP", "SP", "SP", "SP", "SP", "SP"],
        mask_pred_mode=["SP", "SP", "SP", "SP", "P", "P", "P"],
        bbox_flag=True,
        # SegDINO3D-style box-modulated cross-attn in 3D (SP-domain)
        box3d_ca3d=dict(
            enable=True,
            layers='all',
            use_modulation=True,
            temperature=_box3d_pe_temperature,
        ),
    ),
    # Compute GT 3D bboxes/centers/sizes on-the-fly in the SAME 3D space as decoder.
    train_cfg=dict(
        compute_gt_bboxes_3d=True,
        gdino_daca2d=dict(
            # only override the mask threshold/metric; keep other settings from A1.
            # Explicitly pin to SP-domain + paper order to avoid ambiguity if base changes.
            mask=dict(metric=_daca_metric, thr=_daca_thr, domain='sp', order='paper'),
        ),
    ),
    test_cfg=dict(
        gdino_daca2d=dict(
            mask=dict(metric=_daca_metric, thr=_daca_thr, domain='sp', order='paper'),
        ),
    ),
    criterion=dict(
        inst_criterion=dict(
            # SegDINO3D-style matcher: add center/size L1 costs.
            matcher=dict(
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                    dict(type='CenterL1Cost', weight=0.5),
                    dict(type='SizeL1Cost', weight=0.5),
                ],
                topk=1,
            ),
            # [cls, bce, dice, score, bbox_iou, center_l1, size_l1]
            # score is typically None in this codepath, keep its weight 0.
            loss_weight=[0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5],
        ),
    ),
)

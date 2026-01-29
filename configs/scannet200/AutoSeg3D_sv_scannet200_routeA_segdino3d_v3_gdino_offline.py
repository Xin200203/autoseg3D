_base_ = ['./AutoSeg3D_sv_scannet200.py']

# Keep custom modules.
custom_imports = dict(imports=['oneformer3d'])

num_instance_classes = 1
num_semantic_classes = 200
num_instance_classes_eval = 1
voxel_size = 0.02

# --- SegDINO3D-style: 256d point-level early fusion + object-level DACA-2D ---
gdino_target_size = (420, 560)  # (H,W), keep ratio with ScanNet 480x640
gdino_in_dim = 256
gdino_out_dim = 256  # feed 256d into MinkUNet (in_channels = 3+256)

model = dict(
    # SV (single-frame) baseline class: no online memory, no mapping/tracking.
    type='ScanNet200MixFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=voxel_size,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,

    backbone=dict(
        type='Res16UNet34C',
        in_channels=3 + gdino_out_dim,
        out_channels=96,
        config=dict(dilations=[1, 1, 1, 1], conv1_kernel_size=5, bn_momentum=0.02)),
    pool=dict(type='GeoAwarePooling', channel_proj=96),
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        num_layers=6,
        share_attn_mlp=False,
        share_mask_mlp=False,
        cross_attn_mode=["", "SP", "SP", "SP", "SP", "SP", "SP"],
        mask_pred_mode=["SP", "SP", "SP", "SP", "P", "P", "P"],
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=96,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        temporal_attn=False,
        bbox_flag=True,
        # SegDINO3D-style box-modulated CA-3D (optional).
        box3d_ca3d=dict(
            enable=True,
            layers=[0],
            use_modulation=True,
            temperature=10000.0,
        )),
    criterion=dict(
        type='ScanNetMixedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.5),
        inst_criterion=dict(
            type='MixedInstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                    dict(type='CenterL1Cost', weight=0.5),
                    dict(type='SizeL1Cost', weight=0.5)],
                topk=1),
            bbox_loss=dict(type='AxisAlignedIoULoss'),
            # [cls, bce, dice, score, bbox_iou, center_l1, size_l1]
            loss_weight=[0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),

    gdino_backbone=dict(
        type='GroundingDINOBackbone',
        repo_dir='/home/nebula/xxy/GroundingDINO',
        config_path='/home/nebula/xxy/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py',
        checkpoint='/home/nebula/xxy/dataset/models/groundingdino_swinb_cogcoor.pth',
        device='cuda',
        caption='object.',
    ),
    gdino_point_fusion=dict(
        enable=True,
        in_dim=gdino_in_dim,
        out_dim=gdino_out_dim,
        proj_type='identity',
        feat_level=0,
        img_size=gdino_target_size,
        frame_stride=1,
        max_frames=0,
        max_depth=10.0,
        align_corners=False,
        strict=True,
        strict_valid_ratio=0.95,
        log_fail=True,
        log_valid_every=50,
        gdino=dict(backbone=dict(type='GroundingDINOBackbone')),
    ),

    train_cfg=dict(
        compute_gt_bboxes_3d=True,
        gdino_daca2d=dict(
            enable=True,
            img_size=gdino_target_size,
            max_depth=10.0,
            score_thr=0.15,
            max_queries=50,
            query3d_center=dict(min_support_pts=30, reduce='median'),
            strict=True,
            strict_valid_ratio=0.90,
            log_valid_every=50,
            log_first=True,
            order='paper',
            inject_domain='sp',
            inject_layers='auto_sp',
            mask=dict(thr=0.3, metric='l1'),
        ),
    ),
    test_cfg=dict(
        topk_insts=100,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1],
        gdino_daca2d=dict(
            enable=True,
            img_size=gdino_target_size,
            max_depth=10.0,
            score_thr=0.15,
            max_queries=50,
            query3d_center=dict(min_support_pts=30, reduce='median'),
            strict=False,
            strict_valid_ratio=0.90,
            log_valid_every=0,
            order='paper',
            inject_domain='sp',
            inject_layers='auto_sp',
            mask=dict(thr=0.3, metric='l1'),
        ),
    ),
)

# Use local scannet200-sv data root (absolute), keep the rest aligned with baseline.
# Keep consistent with AutoSeg3D SV baseline: use repo-local symlink
# `data/scannet200-sv_fast/` -> `/home/nebula/xxy/3D_Reconstruction/data/scannet200-sv`.
data_root = 'data/scannet200-sv_fast/'

# Keep identical color normalization to baseline SV config (AutoSeg3D_sv_scannet200.py).
color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

# IMPORTANT: baseline SV pipeline does not carry `img_paths/poses/cam_info/points_raw`.
# GDINO point-fusion + DACA-2D require these fields, so we must extend the SV
# pipelines while keeping the rest identical to baseline.
gdino_keys_train = [
    'points',
    'points_raw',
    'gt_labels_3d',
    'pts_semantic_mask',
    'pts_instance_mask',
    'sp_pts_mask',
    'gt_sp_masks',
    'elastic_coords',
    'img_paths',
    'poses',
    'cam_info',
]
gdino_keys_test = [
    'points',
    'points_raw',
    'sp_pts_mask',
    'img_paths',
    'poses',
    'cam_info',
]

train_dataloader = dict(
    batch_size=16,
    num_workers=6,
    dataset=dict(
        ann_file='scannet200_sv_oneformer3d_infos_train.pkl',
        data_root=data_root,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D_',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True,
                with_sp_mask_3d=True),
            # Build `img_paths/poses/cam_info` for GDINO modules (T=1 SV).
            dict(type='PrepareSVForOnline', apply_axis_align=False),
            dict(type='BuildCamInfoFromPoses', dataset_type='scannet200'),
            dict(type='ResizeForGDINO', target_size=gdino_target_size),
            dict(type='NormalizeCamInfo', strict=True),
            dict(type='SavePointsForProjection'),

            dict(type='SwapChairAndFloor'),
            dict(type='PointSegClassMapping'),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-3.14, 3.14],
                scale_ratio_range=[0.8, 1.2],
                translation_std=[0.1, 0.1, 0.1],
                shift_height=False),
            dict(
                type='NormalizePointsColor_',
                color_mean=color_mean,
                color_std=color_std),
            dict(
                type='AddSuperPointAnnotations',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False),
            dict(
                type='ElasticTransfrom',
                gran=[6, 20],
                mag=[40, 160],
                voxel_size=voxel_size,
                p=0.5),
            dict(type='Pack3DDetInputs_', dataset_type='scannet200', keys=gdino_keys_train),
        ],
    ))

val_dataloader = dict(
    dataset=dict(
        ann_file='scannet200_sv_oneformer3d_infos_val.pkl',
        data_root=data_root,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D_',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True,
                with_sp_mask_3d=True),
            dict(type='PrepareSVForOnline', apply_axis_align=False),
            dict(type='BuildCamInfoFromPoses', dataset_type='scannet200'),
            dict(type='ResizeForGDINO', target_size=gdino_target_size),
            dict(type='NormalizeCamInfo', strict=True),
            dict(type='SavePointsForProjection'),
            dict(type='SwapChairAndFloor'),
            dict(type='PointSegClassMapping'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='NormalizePointsColor_',
                        color_mean=color_mean,
                        color_std=color_std),
                    dict(
                        type='AddSuperPointAnnotations',
                        num_classes=num_semantic_classes,
                        stuff_classes=[0, 1],
                        merge_non_stuff_cls=False),
                ]),
            dict(type='Pack3DDetInputs_', dataset_type='scannet200', keys=gdino_keys_test),
        ],
    ))
test_dataloader = val_dataloader

_base_ = ['./AutoSeg3D_sv_scannet200.py']

# A1: only add GDINO object-level DACA-2D (no point-level early fusion).
# Keep model type identical to SV baseline: ScanNet200MixFormer3D (offline, no memory).

gdino_ckpt = '/home/nebula/xxy/dataset/models/groundingdino_swinb_cogcoor.pth'
gdino_repo = '/home/nebula/xxy/GroundingDINO'
gdino_cfg = '/home/nebula/xxy/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'
gdino_img_hw = (420, 560)  # (H,W), fixed resize; keep consistent with V1

num_semantic_classes = 200
voxel_size = 0.02

# Keep identical normalization numbers with baseline config.
color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

model = dict(
    gdino_backbone=dict(
        type='GroundingDINOBackbone',
        repo_dir=gdino_repo,
        config_path=gdino_cfg,
        checkpoint=gdino_ckpt,
        device='cuda',
        caption='object.',
    ),
    train_cfg=dict(
        gdino_daca2d=dict(
            enable=True,
            mode='fuse',
            score_thr=0.10,
            max_queries=50,
            max_depth=10.0,
            log_valid_every=50,
            strict=False,
            log_fail=True,
            mask=dict(metric='l1', thr=0.2, domain='sp', order='paper'),
        ),
    ),
    test_cfg=dict(
        gdino_daca2d=dict(
            enable=True,
            mode='fuse',
            score_thr=0.10,
            max_queries=50,
            max_depth=10.0,
            log_valid_every=1,
            strict=False,
            log_fail=True,
            mask=dict(metric='l1', thr=0.2, domain='sp', order='paper'),
        ),
    ),
)

# -----------------------------------------------------------------------------
# SV pipeline: add 2D-3D projection metadata + points_raw caching (no change to 3D aug).
# -----------------------------------------------------------------------------
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='SavePointsForProjection'),
    dict(type='PrepareSVForOnline', apply_axis_align=True),
    dict(type='BuildCamInfoFromPoses', dataset_type='scannet200'),
    dict(type='ResizeForGDINO', target_size=gdino_img_hw),
    dict(type='NormalizeCamInfo', strict=True),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
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
    dict(
        type='Pack3DDetInputs_SVOnline',
        keys=[
            'points', 'points_raw', 'gt_labels_3d', 'pts_semantic_mask',
            'pts_instance_mask', 'sp_pts_mask', 'gt_sp_masks', 'elastic_coords',
            'img_paths', 'poses', 'cam_info',
        ],
        dataset_type='scannet200'),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='SavePointsForProjection'),
    dict(type='PrepareSVForOnline', apply_axis_align=True),
    dict(type='BuildCamInfoFromPoses', dataset_type='scannet200'),
    dict(type='ResizeForGDINO', target_size=gdino_img_hw),
    dict(type='NormalizeCamInfo', strict=True),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
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
            dict(
                type='Pack3DDetInputs_SVOnline',
                keys=[
                    'points', 'points_raw', 'gt_labels_3d', 'pts_semantic_mask',
                    'pts_instance_mask', 'sp_pts_mask', 'gt_sp_masks',
                    'img_paths', 'poses', 'cam_info',
                ],
                dataset_type='scannet200'),
        ])
]

# IMPORTANT: baseline `train_dataloader` is constructed with the *base* `train_pipeline`
# during config evaluation, so overriding `train_pipeline` alone is NOT enough.
# We must explicitly override dataloaders' pipeline fields for this ablation.
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

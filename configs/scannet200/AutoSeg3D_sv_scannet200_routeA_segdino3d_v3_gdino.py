_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

num_instance_classes = 1
num_semantic_classes = 200
num_instance_classes_eval = 1
voxel_size = 0.02

# --- SegDINO3D-style: 256d point-level early fusion + object-level DACA-2D ---
gdino_target_size = (420, 560)  # (H,W), keep ratio with ScanNet 480x640
gdino_in_dim = 256
gdino_out_dim = 256  # V3: feed 256d into MinkUNet (in_channels = 3+256)

model = dict(
    # Reuse Online model implementation for SV (T=1); disable mapping/tracking.
    type='ScanNet200MixFormer3D_Online',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=voxel_size,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,
    map_to_rec_pcd=False,

    use_query_memory=False,
    use_temporal_loss=False,
    use_decouple=False,
    use_mot=False,
    merge_sp_masks=False,
    debug_mode=False,

    backbone=dict(
        type='Res16UNet34C',
        in_channels=3 + gdino_out_dim,
        out_channels=96,
        config=dict(dilations=[1, 1, 1, 1], conv1_kernel_size=5, bn_momentum=0.02)),
    memory=dict(type='MultilevelMemory', in_channels=[32, 64, 128, 256], queue=-1, vmp_layer=(0, 1, 2, 3)),
    pool=dict(type='GeoAwarePooling', channel_proj=96),
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        # V3 ablation: deepen decoder to test if 2D injection needs more depth.
        num_layers=6,
        share_attn_mlp=False,
        share_mask_mlp=False,
        # length = num_layers + 1 (iter_pred)
        cross_attn_mode=["", "SP", "SP", "SP", "SP", "SP", "SP"],
        # keep SP in early layers (for DACA-2D), refine with P at the end
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
        bbox_flag=True),
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
            # score loss is typically unused when objectness_flag=False; keep weight for backward-compat.
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
        log_missing_bboxes=False,
        gdino_daca2d=dict(
            enable=True,
            # same image resize as point_fusion; relies on pipeline to keep cam_info consistent
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
            mask=dict(thr=0.2, metric='l1'),
        ),
    ),
    test_cfg=dict(
        merge_type='concat',
        topk_insts=100,
        inscat_topk_insts=100,
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
            mask=dict(thr=0.2, metric='l1'),
        ),
    ),
)

dataset_type = 'ScanNet200SegDataset_'
data_root = '/home/nebula/xxy/dataset/data/scannet200-sv/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

# floor and chair are changed
class_names = [
    'wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
    'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer',
    'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor',
    'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
    'music stand', 'projector screen', 'divider', 'laundry detergent',
    'bathroom counter', 'object', 'bathroom vanity', 'closet wall',
    'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin',
    'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case',
    'closet rod', 'coffee kettle', 'structure', 'shower head',
    'keyboard piano', 'case of water bottles', 'coat rack',
    'storage organizer', 'folded chair', 'fire alarm', 'power strip',
    'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
]

color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

train_pipeline = [
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
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor_', color_mean=color_mean, color_std=color_std),
    # IMPORTANT: Online variant returns per-frame lists (len=T), required by Online model loss().
    dict(
        type='AddSuperPointAnnotations_Online',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1],
        merge_non_stuff_cls=False,
        with_rec=False,
        use_sp_gt_ids=False),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=0.5),
    dict(
        type='Pack3DDetInputs_Online',
        dataset_type='scannet200',
        keys=[
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
        ])
]

test_pipeline = [
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
            dict(type='NormalizePointsColor_', color_mean=color_mean, color_std=color_std),
        ]),
    dict(
        type='Pack3DDetInputs_Online',
        dataset_type='scannet200',
        keys=[
            'points',
            'points_raw',
            'gt_labels_3d',
            'pts_semantic_mask',
            'pts_instance_mask',
            'sp_pts_mask',
            'img_paths',
            'poses',
            'cam_info',
        ])
]

train_dataloader = dict(
    # Match AutoSeg3D_sv_scannet200.py hyperparams (may reduce if OOM with 2D fusion enabled).
    batch_size=16,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        ann_file='scannet200_sv_oneformer3d_infos_train.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        ann_file='scannet200_sv_oneformer3d_infos_val.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = val_dataloader

label2cat = {i: name for i, name in enumerate(class_names + ['unlabeled'])}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names + ['unlabeled'])

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112,
    115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139,
    140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169,
    170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230,
    232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331,
    342, 356, 370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609,
    748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191
]
inst_mapping = sem_mapping[2:]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0, 1],
    thing_class_inds=list(range(2, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=128, power=0.9)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

load_from = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=16)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Save only the latest checkpoint and the best one by AP50.
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='all_ap_50%',
        rule='greater',
        save_last=True,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

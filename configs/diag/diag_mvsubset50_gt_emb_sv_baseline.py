_base_ = ['../scannet200/AutoSeg3D_sv_scannet200.py']

# This config is for diagnostics only:
# run a single-frame SV model on MV inputs (multiple frames per scene)
# and enable GT-aligned embedding stability diagnostics (oracle by GT IoU).

custom_imports = dict(imports=['oneformer3d'])

dataset_type = 'ScanNet200SegMVDataset_'
data_root = '/home/nebula/xxy/dataset/data/scannet200-mv_fast'
ann_file = f'{data_root}/scannet200_mv_oneformer3d_infos_val_subset50.pkl'

# Match `AutoSeg3D_scannet200_stage1.py` color normalization for MV data.
color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255,
)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255,
)

# Use the same MV pipeline as stage1/online configs (no training aug, test-time packing).
# Copied from `configs/scannet200/AutoSeg3D_scannet200_stage1.py::test_pipeline`
mv_test_pipeline = [
    dict(
        type='LoadAdjacentDataFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        num_frames=-1,
        num_sample=20000,
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_rec=True,
    ),
    dict(type='SwapChairAndFloorWithRec'),
    dict(type='PointSegClassMappingWithRec'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor_', color_mean=color_mean, color_std=color_std),
            dict(
                type='AddSuperPointAnnotations_Online',
                num_classes=200,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False,
                with_rec=True,
            ),
        ],
    ),
    dict(type='Pack3DDetInputs_Online', keys=['points', 'sp_pts_mask']),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=mv_test_pipeline,
        test_mode=True,
        load_interval=1,
    ),
)

val_dataloader = test_dataloader

# Enable online_monitor output (we reuse UnifiedSegMetric's online_monitor writer)
# and enable GT embedding diagnostics.
model = dict(
    test_cfg=dict(
        online_monitor=dict(
            enable=True,
            gt_vis_npoint=100,
            iou_thr=0.5,
            iou_lo_thr=0.1,
            gt_frame_stride=1,
            gt_emb_diag=dict(
                enable=True,
                frame_stride=1,
                gt_vis_npoint=100,
                iou_thr=0.5,
                iou_lo_thr=0.1,
                min_iou=0.1,
                neg_pairs=2048,
            ),
        ),
    )
)

test_evaluator = dict(
    type='UnifiedSegMetric',
    online_monitor=dict(enable=True, out_dir='online_monitor'),
)

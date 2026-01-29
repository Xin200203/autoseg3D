_base_ = ['./AutoSeg3D_scannet200_stage2.py']

# Finetune track-window STM (identity-safe gating) from a trained stage2 ckpt.
# Keep the experiment small and robust: freeze backbone/neck/memory, warm up
# only STM params, then unfreeze the rest of the head/decoder.

# Start from stage2 checkpoint (not stage1).
load_from = 'work_dirs/AutoSeg3D_scannet200_stage2/epoch_36.pth'

# Track-window STM: use the best inference hyper-params as a starting point.
model = dict(
    asso_config=dict(
        # Keep stage2 training logic (MOT on), but allow decoder grads so the
        # STM branch can be finetuned (stage2 default freezes decoder).
        freeze_decoder=False,
    ),
    decoder=dict(
        track_window_stm=dict(
            enable=True,
            window=5,
            mode='scale',
            dist_lambda=0.5,
            gate_init=-6.0,
        )))

# Short finetune schedule.
max_epochs = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Save best AP checkpoint + always keep latest.
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
        save_best='all_ap',
        rule='greater',
    ),
)

# Slightly smaller LR for finetune.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=max_epochs, power=0.9, by_epoch=True)

# Warmup: only train STM params for 1 epoch, then unfreeze decoder/head.
# Always keep backbone/neck/image-backbone/memory frozen to reduce compute and
# isolate the effect of STM.
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    dict(
        type='FreezeUnfreezeHook',
        warmup_epochs=1,
        warmup_trainable=[
            r'^decoder\._trk_stm_.*',
            r'^decoder\._trk_stm_gate$',
            r'^decoder\.dino_query_cross_attn_layers\..*',
        ],
        always_frozen=[
            r'^backbone\..*',
            r'^neck\..*',
            r'^img_backbone\..*',
            r'^memory\..*',
        ],
        verbose=True,
    ),
]

_base_ = ['./AutoSeg3D_scannet200_stage2_trkstm_finetune.py']

# STM-only finetune (safer): keep the whole segmentation stack frozen and only
# train the newly introduced track-window STM branch + gate. This avoids AP
# drifting due to optimizing association losses with segmentation heads.

max_epochs = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

model = dict(
    asso_config=dict(
        # Allow gradients through decoder forward so STM params can be trained,
        # but keep non-STM parameters frozen via FreezeUnfreezeHook.
        freeze_decoder=False,
    ),
    # Enable online monitor collection during val/test (kept silent unless
    # explicitly enabled here). This writes per-scene JSON + summary under
    # `work_dir/online_monitor_trkstm/` via UnifiedSegMetric.
    test_cfg=dict(
        online_monitor=dict(
            enable=True,
            trk_stm=dict(enable=True),
        )
    ),
    decoder=dict(
        track_window_stm=dict(
            enable=True,
            window=5,
            mode='scale',
            dist_lambda=0.5,
            gate_init=-6.0,
        )))

# Dump aggregated online monitor stats (including trk_stm_apply.*) after val/test.
# Note: config files are evaluated in isolation; use dict-merge semantics to
# override only the `online_monitor` field from the base evaluator.
val_evaluator = dict(online_monitor=dict(enable=True, out_dir="online_monitor_trkstm"))
test_evaluator = dict(online_monitor=dict(enable=True, out_dir="online_monitor_trkstm"))

# Override FreezeUnfreezeHook: never switch to "full" finetune; train only STM.
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    dict(
        type='FreezeUnfreezeHook',
        warmup_epochs=999,
        warmup_trainable=[
            r'^decoder\._trk_stm_.*',
            r'^decoder\._trk_stm_gate$',
        ],
        always_frozen=[
            r'^backbone\..*',
            r'^neck\..*',
            r'^img_backbone\..*',
            r'^memory\..*',
            # Freeze all decoder params except STM branch.
            r'^decoder\.(?!_trk_stm).*',
        ],
        verbose=True,
    ),
]

_base_ = ['./AutoSeg3D_scannet200_stage2_trkstm_finetune_stmonly.py']

# Stage3 (track-window STM) "gate-up" run:
# - Start from a stage2+STM checkpoint (stage3 warm start)
# - Encourage the gate to open: higher init, higher LR, longer schedule
# - Add train-time memory dropout to make STM beneficial (not redundant)

# Warm-start from the best stmonly checkpoint (resets optimizer/scheduler).
load_from = 'work_dirs/_trkstm_ft_w5_l0p5_stmonly_full/best_all_ap_epoch_4.pth'

# Longer schedule (stage2-like length).
max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=4)

model = dict(
    decoder=dict(
        track_window_stm=dict(
            enable=True,
            window=5,
            mode='scale',
            dist_lambda=0.5,
            gate_init=-3.0,   # sigmoid(-3)≈0.047
            mem_dropout=0.2,  # drop memory tokens during training
        )))

# Keep the base finetune LR, but make the gate learn faster.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'decoder._trk_stm_gate': dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = dict(type='PolyLR', begin=0, end=max_epochs, power=0.9, by_epoch=True)


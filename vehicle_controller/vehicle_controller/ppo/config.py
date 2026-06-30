"""
Hyperparameters for the PPO agent.
Set use_recurrence = True to switch to recurrent PPO (LSTM).
"""

CONFIG = {
    # ── Observation dimensions ────────────────────────────────────
    # Image: (C, H, W)
    "img_channels": 2,
    "img_height": 96,
    "img_width":  128,

    # Vector obs: [vx, vy, vz, wx, wy, wz, e_lat, e_lon, e_yaw, e_v, mpc_v, mpc_steer]
    "vec_obs_size": 12,

    # ── Action space ──────────────────────────────────────────────
    # Residual corrections on top of MPC output
    # [delta_v (m/s), delta_steer (rad)]
    "action_size": 2,
    "action_scale": [6.0, 0.0],   # max magnitude of each residual
    "action_bias":  [0.0, 0.0],

    # ── Vehicle ───────────────────────────────────────────────────
    "wheelbase": 2.94,

    # ── PPO hyperparameters ───────────────────────────────────────
    "gamma":  0.99,
    "lamda":  0.95,
    "epochs": 8,
    "n_mini_batch": 8,
    "worker_steps": 1024,          # steps collected before each update
    "value_loss_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "updates":      0,   # stop after N gradient updates (0 = disabled)
    "max_episodes": 200,   # stop after N episodes        (0 = disabled)

    # ── Learning rate schedule ────────────────────────────────────
    "learning_rate_schedule": {
        "initial": 3e-5,
        "final":   1e-6,
        "max_decay_steps": 3_000,
        "power": 1.0,
    },

    # ── Entropy schedule ──────────────────────────────────────────
    "beta_schedule": {
        "initial": 0.001,
        "final":   0.0001,
        "max_decay_steps": 1_000,
        "power": 1.0,
    },

    # ── Clip range schedule ───────────────────────────────────────
    "clip_range_schedule": {
        "initial": 0.1,
        "final":   0.01,
        "max_decay_steps": 1_500,
        "power": 1.0,
    },

    # ── Model ─────────────────────────────────────────────────────
    "hidden_layer_size": 256,
    "cnn_channels": [32, 64, 64],  # conv layer output channels

    # ── Recurrence (flip use_recurrence to True to enable LSTM) ──
    "use_recurrence": True,
    "recurrence": {
        "layer_type": "lstm",       # "gru" or "lstm"
        "hidden_state_size": 256,
        "sequence_length": 32,      # steps per BPTT sequence
        "reset_hidden_state": True, # reset on episode done
    },

    # ── Reward weights ────────────────────────────────────────────
    "reward": {
        "w_lat":   0.1,
        "w_lon":   0.1,
        "w_yaw":   0.01,    
        "w_v":     0.01,
        "w_rev":   0.0,

        "w_roll_rate":  0.1,
        "w_pitch_rate": 0.1,

        "deadband_pitch_deg":  4.0,
        "deadband_roll_deg": 6.0,

        "w_vz": 2.0,

        "w_res_v":     0.005,
        "w_res_steer": 0.00,

        "w_dv":     0.005,
        "w_dsteer": 0.00,

        "w_progress": 50.0,

        "success":        500.0,
        "crash_rollover": -200.0,
        "crash_stuck":    -100.0,
        "crash_fall":     -200.0,
    },

    # ── Checkpoint ────────────────────────────────────────────────
    "save_interval": 100,   # save model every N updates
    "checkpoint_dir": "./checkpoints",
}
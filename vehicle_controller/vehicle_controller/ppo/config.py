"""
Hyperparameters for the PPO agent.
Set use_recurrence = True to switch to recurrent PPO (LSTM).
"""

CONFIG = {
    # ── Observation dimensions ────────────────────────────────────
    # Image: (1, H, W) — mono8 segmentation channel first
    "img_height": 64,
    "img_width":  64,
    # Vector obs: [vx, vy, vz, wx, wy, wz, pos_x, pos_y, yaw,
    #              wp_x, wp_y, wp_yaw, wp_v, mpc_v, mpc_steer]
    "vec_obs_size": 15,

    # ── Action space ─────────────────────────────────────────────
    # Residual corrections on top of MPC output
    # [delta_v (m/s), delta_steer (rad)]
    "action_size": 2,
    "action_scale": [1.0, 0.2],   # max magnitude of each residual
    "action_bias":  [0.0, 0.0],

    # ── PPO hyperparameters ───────────────────────────────────────
    "gamma":  0.99,
    "lamda":  0.95,
    "epochs": 4,
    "n_mini_batch": 4,
    "worker_steps": 256,          # steps collected before each update
    "value_loss_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "updates": 10_000,

    # ── Learning rate schedule ────────────────────────────────────
    "learning_rate_schedule": {
        "initial": 3e-4,
        "final":   1e-5,
        "max_decay_steps": 8_000,
        "power": 1.0,
    },

    # ── Entropy schedule ──────────────────────────────────────────
    "beta_schedule": {
        "initial": 0.01,
        "final":   0.001,
        "max_decay_steps": 8_000,
        "power": 1.0,
    },

    # ── Clip range schedule ───────────────────────────────────────
    "clip_range_schedule": {
        "initial": 0.2,
        "final":   0.1,
        "max_decay_steps": 8_000,
        "power": 1.0,
    },

    # ── Model ─────────────────────────────────────────────────────
    "hidden_layer_size": 256,
    "cnn_channels": [32, 64, 64],  # conv layer output channels

    # ── Recurrence (flip use_recurrence to True to enable LSTM) ──
    "use_recurrence": False,
    "recurrence": {
        "layer_type": "lstm",       # "gru" or "lstm"
        "hidden_state_size": 256,
        "sequence_length": 16,      # steps per BPTT sequence
        "reset_hidden_state": True, # reset on episode done
    },

    # ── Reward weights ────────────────────────────────────────────
    "reward": {
        "w_lat":    1.0,    # lateral error penalty
        "w_lon":    0.3,    # longitudinal error penalty
        "w_yaw":    0.8,    # yaw error penalty
        "w_v":      0.2,    # velocity error penalty
        "w_roll":   2.0,    # roll acceleration penalty (speedbumps)
        "w_pitch":  2.0,    # pitch acceleration penalty (speedbumps)
        "w_action": 0.05,    # action magnitude regularization
        "success":  50.0,   # bonus for completing the route
        "crash":   -50.0,   # penalty for crash / fall
    },

    # ── Checkpoint ────────────────────────────────────────────────
    "save_interval": 100,   # save model every N updates
    "checkpoint_dir": "./checkpoints",
}
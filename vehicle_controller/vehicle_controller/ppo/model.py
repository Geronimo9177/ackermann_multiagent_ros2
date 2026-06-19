"""
Actor-Critic model for continuous residual control.

Architecture:
  ┌─────────────┐   ┌──────────────┐
  │  CNN (img)  │   │  Linear(vec) │
  └──────┬──────┘   └──────┬───────┘
         └────────┬─────────┘
               concat
                 │
          [optional LSTM/GRU]
                 │
            hidden FC
            ┌───┴───┐
         policy   value
         (Normal)  (scalar)

Switching to recurrent:  set config["use_recurrence"] = True.
Everything else (buffer, trainer) stays the same — the recurrent
cell is always passed through; when not using recurrence it is
simply ignored (set to None).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCriticModel(nn.Module):

    LOG_STD_MIN = -4.0
    LOG_STD_MAX =  0.5

    def __init__(self, config: dict):
        super().__init__()

        self.use_recurrence  = config["use_recurrence"]
        self.recurrence_cfg  = config["recurrence"]
        self.hidden_size     = config["hidden_layer_size"]
        self.action_size     = config["action_size"]
        self.action_scale    = torch.tensor(config["action_scale"], dtype=torch.float32)
        self.action_bias     = torch.tensor(config["action_bias"],  dtype=torch.float32)

        img_h   = config["img_height"]
        img_w   = config["img_width"]
        vec_dim = config["vec_obs_size"]
        cnn_ch  = config["cnn_channels"]       # [32, 64, 64]

        # ── CNN (segmentation image) ──────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(1,       cnn_ch[0], kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(cnn_ch[0], cnn_ch[1], kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(cnn_ch[1], cnn_ch[2], kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            cnn_out = int(np.prod(self.cnn(dummy).shape[1:]))
        self.cnn_out_size = cnn_out

        # Orthogonal init for conv layers
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

        # ── Vector encoder ────────────────────────────────────────
        self.vec_encoder = nn.Sequential(
            nn.Linear(vec_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
        )
        nn.init.orthogonal_(self.vec_encoder[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.vec_encoder[2].weight, np.sqrt(2))

        fused_size = cnn_out + 128

        # ── Optional recurrent layer ──────────────────────────────
        rnn_hidden = self.recurrence_cfg["hidden_state_size"]
        if self.use_recurrence:
            layer_type = self.recurrence_cfg["layer_type"]
            if layer_type == "gru":
                self.rnn = nn.GRU(fused_size, rnn_hidden, batch_first=True)
            elif layer_type == "lstm":
                self.rnn = nn.LSTM(fused_size, rnn_hidden, batch_first=True)
            else:
                raise ValueError(f"Unknown recurrence layer type: {layer_type}")
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.orthogonal_(param, np.sqrt(2))
            fc_in = rnn_hidden
        else:
            self.rnn = None
            fc_in = fused_size

        # ── Shared hidden ─────────────────────────────────────────
        self.fc_shared = nn.Linear(fc_in, self.hidden_size)
        nn.init.orthogonal_(self.fc_shared.weight, np.sqrt(2))

        # ── Policy head ───────────────────────────────────────────
        self.fc_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.fc_policy.weight, np.sqrt(2))

        self.mean_head    = nn.Linear(self.hidden_size, self.action_size)
        self.log_std_head = nn.Linear(self.hidden_size, self.action_size)
        nn.init.orthogonal_(self.mean_head.weight,    0.01)
        nn.init.orthogonal_(self.log_std_head.weight, 0.01)

        # ── Value head ────────────────────────────────────────────
        self.fc_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.fc_value.weight, np.sqrt(2))

        self.value_head = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, img: torch.Tensor, vec: torch.Tensor,
                recurrent_cell=None, sequence_length: int = 1):
        """
        Args:
            img:            (B, 1, H, W)  — segmentation image, float [0,1]
            vec:            (B, vec_dim)  — vector observations
            recurrent_cell: hidden state (or None if not using recurrence)
            sequence_length: used during BPTT; 1 during sampling

        Returns:
            dist:   Normal distribution over residual actions
            value:  (B,)
            recurrent_cell: updated hidden state (or None)
        """
        B = img.size(0)

        # CNN branch
        h_img = self.cnn(img).reshape(B, -1)

        # Vector branch
        h_vec = self.vec_encoder(vec)

        # Fuse
        h = torch.cat([h_img, h_vec], dim=-1)

        # Optional RNN
        if self.use_recurrence and self.rnn is not None:
            if sequence_length == 1:
                h, recurrent_cell = self.rnn(h.unsqueeze(1), recurrent_cell)
                h = h.squeeze(1)
            else:
                h = h.reshape(B // sequence_length, sequence_length, -1)
                h, recurrent_cell = self.rnn(h, recurrent_cell)
                h = h.reshape(B, -1)
        else:
            recurrent_cell = None

        # Shared hidden
        h = F.relu(self.fc_shared(h))

        # Policy
        h_pi  = F.relu(self.fc_policy(h))
        mean  = torch.tanh(self.mean_head(h_pi))          # raw in [-1,1]
        log_std = self.log_std_head(h_pi).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std   = log_std.exp()

        # Scale mean to action range
        scale  = self.action_scale.to(img.device)
        bias   = self.action_bias.to(img.device)
        dist   = Normal(mean * scale + bias, std * scale)

        # Value
        h_v   = F.relu(self.fc_value(h))
        value = self.value_head(h_v).reshape(-1)

        return dist, value, recurrent_cell

    def init_recurrent_cell_states(self, batch_size: int, device: torch.device):
        """Returns zero-initialized hidden states."""
        if not self.use_recurrence:
            return None, None
        h = self.recurrence_cfg["hidden_state_size"]
        hxs = torch.zeros(1, batch_size, h, device=device)
        cxs = torch.zeros(1, batch_size, h, device=device) \
              if self.recurrence_cfg["layer_type"] == "lstm" else None
        return hxs, cxs
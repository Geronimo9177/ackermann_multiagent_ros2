"""
Replay buffer for on-policy PPO with a single ROS-based environment.
"""

import numpy as np
import torch

class Buffer:
    def __init__(self, config: dict, device: torch.device) -> None:
        self.device        = device
        self.worker_steps  = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size    = self.worker_steps          # n_workers == 1
        self.mini_batch_size = self.batch_size // self.n_mini_batches

        self.use_recurrence = config["use_recurrence"]
        rec_cfg            = config["recurrence"]
        self.layer_type    = rec_cfg["layer_type"]
        self.sequence_length = rec_cfg["sequence_length"]
        hidden_size        = rec_cfg["hidden_state_size"]

        img_ch = config.get("img_channels", 2)
        img_h   = config["img_height"]
        img_w   = config["img_width"]
        vec_dim = config["vec_obs_size"]
        act_dim = config["action_size"]

        T = self.worker_steps

        # ── Storage ───────────────────────────────────────────────
        self.imgs = torch.zeros((T, img_ch, img_h, img_w),  dtype=torch.float32)
        self.vecs     = torch.zeros((T, vec_dim),           dtype=torch.float32)
        self.actions  = torch.zeros((T, act_dim),           dtype=torch.float32)
        self.log_probs = torch.zeros((T, act_dim),          dtype=torch.float32)
        self.values   = torch.zeros(T,                      dtype=torch.float32)
        self.rewards  = torch.zeros(T,                      dtype=torch.float32)
        self.dones    = torch.zeros(T,                      dtype=torch.bool)
        self.advantages = torch.zeros(T,                    dtype=torch.float32)

        # Recurrent states (stored at each step for BPTT)
        self.hxs = torch.zeros((T, hidden_size), dtype=torch.float32)
        self.cxs = torch.zeros((T, hidden_size), dtype=torch.float32)

        self.step = 0   # current write index

    # ── Write ─────────────────────────────────────────────────────

    def store(self, img, vec, action, log_prob, value, reward, done,
              hx=None, cx=None):
        """Store one transition. Call after each env step."""
        t = self.step
        self.imgs[t]      = img.cpu()
        self.vecs[t]      = vec.cpu()
        self.actions[t]   = action.cpu()
        self.log_probs[t] = log_prob.cpu()
        self.values[t]    = value.cpu()
        self.rewards[t]   = reward
        self.dones[t]     = done
        if hx is not None:
            self.hxs[t] = hx.squeeze().cpu()
        if cx is not None:
            self.cxs[t] = cx.squeeze().cpu()
        self.step += 1

    def full(self) -> bool:
        return self.step >= self.worker_steps

    def reset_step(self):
        self.step = 0

    # ── GAE ───────────────────────────────────────────────────────

    def calc_advantages(self, last_value: torch.Tensor,
                        gamma: float, lamda: float) -> None:
        with torch.no_grad():
            last_adv   = 0.0
            last_val   = last_value.cpu().item()
            for t in reversed(range(self.worker_steps)):
                mask       = 1.0 - float(self.dones[t].item())
                delta      = (self.rewards[t].item()
                              + gamma * last_val * mask
                              - self.values[t].item())
                last_adv   = delta + gamma * lamda * last_adv * mask
                self.advantages[t] = last_adv
                last_val   = self.values[t].item()

    # ── Mini-batch generator ─────────────────────────────────────

    def mini_batch_generator(self):
        """Yields shuffled mini-batches (non-recurrent path)."""
        indices = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            idx = indices[start : start + self.mini_batch_size]
            yield {
                "imgs":       self.imgs[idx].to(self.device),
                "vecs":       self.vecs[idx].to(self.device),
                "actions":    self.actions[idx].to(self.device),
                "log_probs":  self.log_probs[idx].to(self.device),
                "values":     self.values[idx].to(self.device),
                "advantages": self.advantages[idx].to(self.device),
                "hxs":        None,
                "cxs":        None,
            }

    def recurrent_mini_batch_generator(self, layer_type: str):
        """
        Yields mini-batches that preserve temporal ordering within sequences.
        Used when use_recurrence = True.
        """
        T   = self.worker_steps
        SL  = self.sequence_length

        # Build sequence start indices respecting episode boundaries
        seq_starts = []
        ep_start   = 0
        for t in range(T):
            if t - ep_start >= SL or (self.dones[t] and t < T - 1):
                seq_starts.append(ep_start)
                ep_start = t + 1 if self.dones[t] else t
        seq_starts.append(ep_start)   # last incomplete sequence

        num_seqs = len(seq_starts)
        seq_indices = torch.randperm(num_seqs)
        seqs_per_mb = max(1, num_seqs // self.n_mini_batches)

        for mb_start in range(0, num_seqs, seqs_per_mb):
            mb_seq_idx = seq_indices[mb_start : mb_start + seqs_per_mb]
            imgs_list, vecs_list, act_list, lp_list = [], [], [], []
            val_list, adv_list, hx_list, cx_list    = [], [], [], []
            loss_masks = []

            for si in mb_seq_idx:
                s = seq_starts[si.item()]
                e = min(s + SL, T)
                length = e - s

                pad = SL - length
                def _pad(t):
                    if pad == 0: return t
                    shape = (pad,) + t.shape[1:]
                    return torch.cat([t, torch.zeros(shape, dtype=t.dtype)], 0)

                imgs_list.append(_pad(self.imgs[s:e]))
                vecs_list.append(_pad(self.vecs[s:e]))
                act_list.append(_pad(self.actions[s:e]))
                lp_list.append(_pad(self.log_probs[s:e]))
                val_list.append(_pad(self.values[s:e].unsqueeze(-1)))
                adv_list.append(_pad(self.advantages[s:e].unsqueeze(-1)))

                mask = torch.zeros(SL, dtype=torch.bool)
                mask[:length] = True
                loss_masks.append(mask)

                hx_list.append(self.hxs[s].unsqueeze(0))
                cx_list.append(self.cxs[s].unsqueeze(0))

            def _stack(lst): return torch.stack(lst, 0).to(self.device)

            imgs_b = _stack(imgs_list).reshape(-1, *self.imgs.shape[1:])
            vecs_b = _stack(vecs_list).reshape(-1, self.vecs.shape[-1])
            act_b  = _stack(act_list).reshape(-1, self.actions.shape[-1])
            lp_b   = _stack(lp_list).reshape(-1, self.log_probs.shape[-1])
            mask_b = _stack(loss_masks).reshape(-1)
            hx_b   = torch.stack(hx_list, 0).squeeze(1).unsqueeze(0).to(self.device)
            cx_b   = torch.stack(cx_list, 0).squeeze(1).unsqueeze(0).to(self.device) \
                     if layer_type == "lstm" else None

            # Values and advantages: only unpadded entries
            val_b  = _stack(val_list).reshape(-1)[mask_b]
            adv_b  = _stack(adv_list).reshape(-1)[mask_b]

            yield {
                "imgs":        imgs_b,
                "vecs":        vecs_b,
                "actions":     act_b,
                "log_probs":   lp_b[mask_b],
                "values":      val_b,
                "advantages":  adv_b,
                "loss_mask":   mask_b,
                "hxs":         hx_b,
                "cxs":         cx_b,
                "seq_length":  SL,
            }
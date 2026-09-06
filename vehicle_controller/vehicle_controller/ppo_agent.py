#!/usr/bin/env python3
"""
PPO agent node — integrates with the existing MPC + Webots pipeline.

Topic graph (additions marked with ★):
PPO agent (reads /cmd_vel AFTER MPC publishes)
    ├─ obs: /camera/segmentation, /ground_truth_odom,
    ├─ publishes★ /cmd_vel (overrides MPC with base + residual)
    └─ reward★: /ppo/reward (from rl_master)

Timing contract (training mode):
  1. Webots fires /sim/trigger  →  sim is PAUSED
  2. MPC computes and publishes /cmd_vel_mpc
  3. PPO waits for /cmd_vel_mpc, builds obs, runs forward pass, publishes /cmd_vel (base + residual)
  4. CarDriver sees new /cmd_vel stamp → resumes sim
"""

import os
import math
import time
import threading
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64MultiArray, Bool, String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from tf_transformations import euler_from_quaternion
from torch.utils.tensorboard import SummaryWriter

from .ppo.config import CONFIG
from .ppo.model import ActorCriticModel
from .ppo.buffer import Buffer


def _polynomial_decay(initial, final, max_steps=None, power=1.0, step=0, max_decay_steps=None):
    if max_steps is None:
        max_steps = max_decay_steps
    if max_steps is None:
        raise ValueError('max_steps or max_decay_steps must be provided')
    if step >= max_steps:
        return final
    return (initial - final) * ((1 - step / max_steps) ** power) + final


class PPOAgentNode(Node):
    """
    ROS 2 node that wraps the PPO training loop.
    """

    def __init__(self):
        super().__init__('ppo_agent')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('training_mode',    True)
        self.declare_parameter('use_ground_truth', False)
        self.declare_parameter('run_id',           'ppo_run')
        self.declare_parameter('checkpoint_dir',   CONFIG['checkpoint_dir'])

        self.training_mode    = self.get_parameter('training_mode').value
        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        self.run_id           = self.get_parameter('run_id').value
        self.ckpt_dir         = self.get_parameter('checkpoint_dir').value

        self.config = CONFIG
        self.cfg_rec = CONFIG['recurrence']
        self.use_rec = CONFIG['use_recurrence']

        # ── Device ────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'PPO using device: {self.device}')

        # ── Model & optimizer ─────────────────────────────────────
        self.model = ActorCriticModel(self.config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate_schedule']['initial']
        )

        # Recurrent cell state (batch_size = 1 during rollout)
        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
        self.recurrent_cell = (hxs, cxs) if (self.use_rec and cxs is not None) \
                              else hxs

        # ── Buffer ────────────────────────────────────────────────
        self.buffer  = Buffer(self.config, self.device)

        # ── Tensorboard ───────────────────────────────────────────
        os.makedirs('./summaries', exist_ok=True)
        ts = time.strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'./summaries/{self.run_id}/{ts}')

        # ── Training counters ─────────────────────────────────────
        self.update_count     = 0
        self.episode_count    = 0
        self._ep_reward       = 0.0
        self._ep_length       = 0
        self._control_step = 0

        # ── Observation cache ─────────────────────────────────────
        # These are updated asynchronously by subscriber callbacks
        self._img_msg:   Image       = None
        self._odom_msg:  Odometry    = None
        self._mpc_cmd:   TwistStamped = None
        self._trajectory_id = ''
        self._lock       = threading.Lock()

        self.bridge = CvBridge()
        self._blank_img = np.zeros((self.config.get('img_channels', 2), self.config['img_height'], self.config['img_width']), dtype=np.float32)

        self._e_lat = 0.0
        self._e_lon = 0.0
        self._e_yaw = 0.0
        self._e_v   = 0.0
        self._v_z   = 0.0

        # Memory of final control signal commands for Slew Rate tracking
        self._dv                    = 0.0
        self._dsteer                = 0.0
        self._prev_final_v          = 0.0
        self._prev_final_steer      = 0.0
        self._prev_prev_final_v     = 0.0
        self._prev_prev_final_steer = 0.0
        self._prev_track_progress   = 0.0

        self._last_v_cmd_mpc = 0.0
        self._last_steer_mpc = 0.0

        self._cached_reward = {
            'total': 0.0, 'lat': 0.0, 'lon': 0.0, 'yaw': 0.0,
            'v': 0.0, 'slew': 0.0, 'rates_gated': 0.0, 'vz': 0.0,
            'terminal': 0.0
        }

        # ── Previous action for the buffer ────────────────────────
        self._prev_action   = torch.zeros(self.config['action_size'])
        self._prev_log_prob = torch.zeros(self.config['action_size'])
        self._prev_value    = torch.zeros(1)
        self._prev_img      = None
        self._prev_vec      = None
        self._have_prev     = False

        # ── Action scaling ────────────────────────────────────────
        self._action_scale = torch.tensor(self.config['action_scale'], dtype=torch.float32)
        self._action_bias  = torch.tensor(self.config.get('action_bias'), dtype=torch.float32)

        # ── Checkpoint ────────────────────────────────────────────
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._try_load_checkpoint()

        # Background training state
        self._training = False

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Image,       '/camera/segmentation',
                                 self._img_cb,  10)
        
        self.create_subscription(Int32,       '/rl/result',
                                 self._result_cb,  1)

        self.create_subscription(String,      '/rl/trajectory_id',
                     self._trajectory_id_cb, 1)
        
        self.create_subscription(Float64MultiArray, '/mpc/debug',
                                self._mpc_debug_cb, 10)

        odom_topic = '/ground_truth_odom' if self.use_ground_truth \
                     else '/odometry/fused'
        self.create_subscription(Odometry, odom_topic,
                                 self._odom_cb, 10)

        # ── Publishers ────────────────────────────────────────────
        self.cmd_pub            = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.ep_met_pub         = self.create_publisher(Float64MultiArray, '/ppo/episode_metrics', 10)
        self.train_met_pub      = self.create_publisher(Float64MultiArray, '/ppo/train_metrics', 10)
        self.reward_terms_pub   = self.create_publisher(Float64MultiArray, '/ppo/reward_terms', 10)

        self.get_logger().info('PPO agent ready.')

    # ═══════════════════════════════════════════════════════════════
    # Subscriber callbacks
    # ═══════════════════════════════════════════════════════════════

    def _img_cb(self, msg: Image):
        with self._lock:
            self._img_msg = msg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom_msg = msg

    def _trajectory_id_cb(self, msg: String):
        with self._lock:
            self._trajectory_id = msg.data
    
    def _mpc_debug_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 11:
            return
        
        x_now, y_now, yaw_now = msg.data[0], msg.data[1], msg.data[2]
        x_ref, y_ref, yaw_ref = msg.data[3], msg.data[4], msg.data[5]
        v_cmd   = float(msg.data[6])
        v_ref   = float(msg.data[7])  
        steer   = float(msg.data[8])

        track_progress = float(msg.data[9])

        self._last_v_cmd_mpc = v_cmd
        self._last_steer_mpc = steer

        # Positional errors in the Frenet frame
        dx = x_now - x_ref
        dy = y_now - y_ref
        e_lat = -math.sin(yaw_ref) * dx + math.cos(yaw_ref) * dy
        e_lon = math.cos(yaw_ref) * dx + math.sin(yaw_ref) * dy
        e_yaw = math.atan2(math.sin(yaw_now - yaw_ref), math.cos(yaw_now - yaw_ref))

        with self._lock:
            odom_msg = self._odom_msg
        
        v_x, e_v, v_z, roll_w_deg, pitch_w_deg = 0.0, 0.0, 0.0, 0.0, 0.0
        if odom_msg is not None:
                v_x = odom_msg.twist.twist.linear.x
                v_now       = math.hypot(odom_msg.twist.twist.linear.x,
                                     odom_msg.twist.twist.linear.y)
                e_v         = v_now - v_ref
                v_z         = odom_msg.twist.twist.linear.z
                roll_w_deg  = math.degrees(abs(odom_msg.twist.twist.angular.x))
                pitch_w_deg = math.degrees(abs(odom_msg.twist.twist.angular.y))

        # Slew Rate 
        dv_action = self._prev_final_v - self._prev_prev_final_v
        dsteer_action = self._prev_final_steer - self._prev_prev_final_steer

        # Progress
        dp = track_progress - self._prev_track_progress
        self._prev_track_progress = track_progress

        # Angular velocity
        w = self.config['reward']
        pw_gated = pitch_w_deg if pitch_w_deg > w['deadband_pitch_deg'] else 0.0
        rw_gated = roll_w_deg if roll_w_deg > w['deadband_roll_deg']  else 0.0
            
        # Term reward
        r_lat   = -w['w_lat']        * (e_lat ** 2)
        r_lon   = -w['w_lon']        * (e_lon ** 2)
        r_yaw   = -w['w_yaw']        * (e_yaw ** 2)
        r_v     = -w['w_v']          * (e_v   ** 2)     - w['w_rev']   * max(0.0, -v_x) 
        r_res   = -w['w_res_v']      * (self._dv ** 2)  - w['w_res_steer'] * (self._dsteer ** 2)
        r_slew  = -w['w_dv']         * (dv_action ** 2) - w['w_dsteer'] * (dsteer_action ** 2)
        r_rates = -w['w_pitch_rate'] * pw_gated         - w['w_roll_rate'] * rw_gated
        r_prog  =  w['w_progress']   * (dp * 100)
        r_vz    = -w['w_vz']         * (v_z**2)

        with self._lock:
            self._e_lat, self._e_lon = e_lat, e_lon
            self._e_yaw, self._e_v   = e_yaw, e_v

            self._cached_reward = {
                'total':       r_lat + r_lon + r_yaw + r_v + r_res + r_slew + r_rates + r_prog + r_vz,
                'lat':         r_lat,
                'lon':         r_lon,
                'yaw':         r_yaw,
                'v':           r_v,
                'r_res':       r_res,
                'slew':        r_slew,
                'rates_gated': r_rates,
                'progress':    r_prog,
                'vz':          r_vz,
                'terminal':    0.0,
            }

        self._main_loop(v_cmd, steer)
        
    def _result_cb(self, msg: Int32):
        """Episode result from rl_master (0=running,1=success,2=crash,3=fall)."""
        result = msg.data
        if result == 0:
            return

        w = self.config['reward']
        if result == 1:
            terminal = w['success']
        elif result == 2:    # ROLLOVER 
            terminal = w['crash_rollover']
        elif result == 3:    # STUCK
            terminal = w['crash_stuck']
        elif result == 4:    # FALL
            terminal = w['crash_fall']
        else:
            terminal = 0.0
        
        with self._lock:
            self._cached_reward['terminal'] = terminal
            self._cached_reward['total']   += terminal
        
        if self._have_prev and self.training_mode:
            self._store_transition(done=True)

        self.reset()

    def reset(self):
        """Reset recurrent state and episode tracking."""
        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
        self.recurrent_cell = (hxs, cxs) if (self.use_rec and cxs is not None) else hxs
        self._have_prev = False
        self._ep_reward = 0.0
        self._ep_length = 0
        
        self._prev_final_v          = 0.0
        self._prev_final_steer      = 0.0
        self._prev_prev_final_v     = 0.0
        self._prev_prev_final_steer = 0.0
        self._prev_track_progress   = 0.0
        self._dv                    = 0.0      
        self._dsteer                = 0.0
        self._last_v_cmd_mpc        = 0.0
        self._last_steer_mpc        = 0.0

        with self._lock:
            self._e_lat = 0.0
            self._e_lon = 0.0
            self._e_yaw = 0.0
            self._e_v   = 0.0
            self._cached_reward['terminal'] = 0.0
            self._cached_reward['total']    = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Main control loop (called once per sim step, sim is PAUSED)
    # ═══════════════════════════════════════════════════════════════

    def _main_loop(self, v_cmd_mpc: float, steer_mpc: float):
        """
        1. Build current observation
        2. Store previous transition (obs, action, log_prob, value, reward)
        3. Run forward pass → publish /cmd_vel
        4. Cache (obs, action, log_prob, value) for next step
        5. If buffer is full → train
        """
        # ── Snapshot observations ──────────────────────────────
        with self._lock:
            img_msg  = self._img_msg
            odom_msg = self._odom_msg
            e_lat, e_lon = self._e_lat, self._e_lon
            e_yaw, e_v   = self._e_yaw, self._e_v

        img_t, vec_t = self._build_obs(img_msg, odom_msg, v_cmd_mpc, steer_mpc,
                                       e_lat, e_lon, e_yaw, e_v)

        # ── Store previous transition ──────────────────────────
        if self._have_prev and self.training_mode:
            self._store_transition(done=False)

        # ── Forward pass ───────────────────────────────────────
        with torch.no_grad():
            img_d = img_t.unsqueeze(0).to(self.device)
            vec_d = vec_t.unsqueeze(0).to(self.device)

            dist, value, self.recurrent_cell = self.model(
                img_d, vec_d, self.recurrent_cell
            )

            if self.training_mode:
                action = dist.sample()
            else:
                action = dist.mean             # deterministic at test time

            log_prob = dist.log_prob(action)  # (1, action_size)

        # ── Publish /cmd_vel (MPC base + PPO residual) ─────────
        self._publish_cmd(v_cmd_mpc, steer_mpc, action.squeeze(0))

        # ── Cache for next step ────────────────────────────────
        self._prev_img      = img_t
        self._prev_vec      = vec_t
        self._prev_action   = action.squeeze(0).cpu()
        self._prev_log_prob = log_prob.squeeze(0).cpu()
        self._prev_value    = value.squeeze(0).cpu()
        self._have_prev     = True
    
    def _store_transition(self, done: bool):
        """Helper function to store transitions, publish telemetry, and trigger training."""
        with self._lock:
            terms = dict(self._cached_reward)
        reward = terms['total']

        reward_terms_msg = Float64MultiArray()
        reward_terms_msg.data = [
            float(self._control_step), float(terms['total']),
            float(terms['lat']),   float(terms['lon']),    float(terms['yaw']),
            float(terms['v']),     float(terms['slew']),   float(terms['rates_gated']),
            float(terms['vz']),    float(terms['r_res']),  float(terms['progress']),
            float(terms['terminal'])
        ]
        self.reward_terms_pub.publish(reward_terms_msg)
        self._control_step += 1

        hx, cx = self._cell_arrays()
        self.buffer.store(
            img=self._prev_img, vec=self._prev_vec,
            action=self._prev_action, log_prob=self._prev_log_prob,
            value=self._prev_value,   reward=reward,
            done=done, hx=hx, cx=cx,
        )
        self._ep_reward += reward
        self._ep_length += 1

        if done:
            self.episode_count += 1

            ep_msg = Float64MultiArray()
            ep_msg.data = [
                float(self.episode_count),
                float(self._ep_reward),
                float(self._ep_length),
                float(self.update_count),
                float(self._control_step),   # total env steps at episode end
            ]
            self.ep_met_pub.publish(ep_msg)

            with self._lock:
                trajectory_id = self._trajectory_id
            self.get_logger().info(
                f'[Ep {self.episode_count}] route={trajectory_id or "unknown"}')

            self.writer.add_scalar('episode_raw/reward', self._ep_reward, self._control_step)
            self.writer.add_scalar('episode_raw/length', self._ep_length, self._control_step)

            max_ep = self.config.get('max_episodes', 0)
            if max_ep > 0 and self.episode_count >= max_ep:
                self.get_logger().info(
                    f'Reached max_episodes={max_ep}. Saving and stopping.')
                self._save_checkpoint(self.episode_count)
                self.writer.close()
                rclpy.try_shutdown()
                return

            save_every = self.config.get('save_interval_episodes', 10)
            if save_every > 0 and self.episode_count % save_every == 0:
                self._save_checkpoint(self.episode_count)

        if self.buffer.full():
            self._train()
            self.buffer.reset_step()

    # ═══════════════════════════════════════════════════════════════
    # Observation builder
    # ═══════════════════════════════════════════════════════════════

    def _build_obs(self, img_msg, odom_msg,
                   v_cmd_mpc: float, steer_mpc: float,
                   e_lat=0.0, e_lon=0.0, e_yaw=0.0, e_v=0.0):
        """Returns (img_tensor [1,H,W], vec_tensor [vec_dim])."""
        cfg = self.config
        H, W = cfg['img_height'], cfg['img_width']

        # ── Image ─────────────────────────────────────────────────
        if img_msg is not None:
            raw = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            raw = raw.astype(np.float32) / 255.0
            raw = raw.transpose(2, 0, 1) # [H, W, C] -> [C, H, W]
        else:
            raw = self._blank_img.copy()

        img_t = torch.from_numpy(raw)

        # ── Vector obs ────────────────────────────────────────────
        # [vx, vy, vz, wx, wy, wz, e_lat, e_lon, e_yaw, e_v, mpc_v, mpc_steer]
        vec = np.zeros(cfg['vec_obs_size'], dtype=np.float32)

        if odom_msg is not None:
            t = odom_msg.twist.twist
            vec[0], vec[1], vec[2] = t.linear.x, t.linear.y, t.linear.z
            vec[3], vec[4], vec[5] = t.angular.x, t.angular.y, t.angular.z
            
        vec[6]  = float(e_lat)
        vec[7]  = float(e_lon)
        vec[8]  = float(e_yaw)
        vec[9]  = float(e_v)
        vec[10] = float(v_cmd_mpc)
        vec[11] = float(steer_mpc)

        return img_t, torch.from_numpy(vec)

    # ═══════════════════════════════════════════════════════════════
    # Publish
    # ═══════════════════════════════════════════════════════════════

    def _publish_cmd(self, v_cmd_mpc: float, steer_mpc: float, residual: torch.Tensor):
        """Add PPO residual on top of MPC command and publish."""
        # Residual is in [-1, 1] range, scale to actual action range
        dv, d_steer = (residual.cpu() * self._action_scale + self._action_bias).tolist()

        L           = self.config.get('wheelbase', 2.94)
        v_final     = v_cmd_mpc + dv
        steer_final = steer_mpc + d_steer
        w_final     = (v_final / L * math.tan(steer_final)) if abs(v_final) > 0.01 else 0.0

        self._dv = dv
        self._dsteer = d_steer
        self._prev_prev_final_v = self._prev_final_v
        self._prev_prev_final_steer = self._prev_final_steer
        self._prev_final_v = v_final
        self._prev_final_steer = steer_final

        msg = TwistStamped()
        msg.header.stamp   = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v_final)
        msg.twist.angular.z = float(w_final)
        if rclpy.ok():
            self.cmd_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════

    def _train(self):
        t0 = time.perf_counter()

        # Compute last value for GAE bootstrap
        with self._lock:
            img_msg  = self._img_msg
            odom_msg = self._odom_msg
            e_lat, e_lon = self._e_lat, self._e_lon
            e_yaw, e_v   = self._e_yaw, self._e_v

        img_t, vec_t = self._build_obs(img_msg, odom_msg,
                                       self._last_v_cmd_mpc, self._last_steer_mpc,
                                       e_lat, e_lon, e_yaw, e_v)

        with torch.no_grad():
            _, last_value, _ = self.model(
                img_t.unsqueeze(0).to(self.device),
                vec_t.unsqueeze(0).to(self.device),
                self.recurrent_cell,
            )
        self.buffer.calc_advantages(last_value, self.config['gamma'],
                                    self.config['lamda'])

        cfg   = self.config
        step  = self.update_count
        lr    = _polynomial_decay(**cfg['learning_rate_schedule'], step=step)
        beta  = _polynomial_decay(**cfg['beta_schedule'],          step=step)
        clip  = _polynomial_decay(**cfg['clip_range_schedule'],    step=step)

        all_stats = []
        for _ in range(cfg['epochs']):
            gen = (self.buffer.recurrent_mini_batch_generator(
                       self.cfg_rec['layer_type'])
                   if self.use_rec
                   else self.buffer.mini_batch_generator())
            for mb in gen:
                all_stats.append(self._train_mini_batch(mb, lr, clip, beta))

        stats = np.mean(all_stats, axis=0)  # [pi_loss, v_loss, loss, entropy, approx_kl, mean_log_ratio]
        self.update_count += 1

        # Logging
        total_steps = step * cfg['worker_steps']
        self.writer.add_scalar('losses/policy_loss', stats[0], total_steps)
        self.writer.add_scalar('losses/value_loss',  stats[1], total_steps)
        self.writer.add_scalar('losses/loss',        stats[2], total_steps)
        self.writer.add_scalar('losses/entropy',     stats[3], total_steps)
        self.writer.add_scalar('training/approx_kl', stats[4], total_steps)
        self.writer.add_scalar('training/mean_log_ratio', stats[5], total_steps)
        self.writer.add_scalar('training/lr', lr, total_steps)

        elapsed = time.perf_counter() - t0
        self.get_logger().info(
            f'[Update {step} | Step {total_steps}] loss={stats[2]:.4f} pi={stats[0]:.4f} '
            f'v={stats[1]:.4f} H={stats[3]:.4f} '
            f'KL={stats[4]:.6f} log_ratio={stats[5]:.6f} '
            f'lr={lr:.2e} t={elapsed*1000:.0f}ms'
        )

        train_msg = Float64MultiArray()
        train_msg.data = [
            float(total_steps),         # Total env steps (common scale)
            float(stats[0]),            # Policy Loss
            float(stats[1]),            # Value Loss
            float(stats[2]),            # Total Loss
            float(stats[3]),            # Entropy
            float(lr),                  # Learning Rate
            float(elapsed * 1000.0),    # Time ms
            float(self.episode_count),  # Episode
            float(stats[4]),            # Approximate sampled KL
            float(stats[5])             # Mean log(pi_new / pi_old)
        ]
        self.train_met_pub.publish(train_msg)

        self_max_updates = cfg.get('updates', 0)
        if self_max_updates > 0 and self.update_count >= self_max_updates:
            self.get_logger().info(
                f'Reached max updates={self_max_updates}. Saving and stopping.')
            self._save_checkpoint(self.episode_count)
            self.writer.close()
            rclpy.try_shutdown()

    def _train_mini_batch(self, mb: dict, lr: float, clip: float,
                          beta: float) -> list:
        seq_len = mb.get('seq_length', 1)
        dist, value, _ = self.model(
            mb['imgs'], mb['vecs'],
            recurrent_cell=(mb['hxs'], mb['cxs'])
                            if (self.use_rec and mb['hxs'] is not None) else None,
            sequence_length=seq_len,
        )

        # Apply loss mask if present (recurrent path)
        mask = mb.get('loss_mask', None)

        log_probs  = dist.log_prob(mb['actions'])    # (B, act_dim)
        entropies  = dist.entropy()                  # (B, act_dim)

        if mask is not None:
            value      = value[mask]
            log_probs  = log_probs[mask]
            entropies  = entropies[mask]

        adv = mb['advantages']
        adv_norm = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)

        # Expand adv for multi-dim action
        adv_exp = adv_norm.unsqueeze(-1).expand_as(log_probs)

        old_lp = mb['log_probs']   # already masked on recurrent path
        log_ratio = log_probs - old_lp
        ratio  = torch.exp(log_ratio)
        surr1  = ratio * adv_exp
        surr2  = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_exp
        pi_loss = -torch.min(surr1, surr2).mean()

        ret        = mb['values'] + adv
        v_clipped  = mb['values'] + (value - mb['values']).clamp(-clip, clip)
        v_loss     = torch.max((value - ret)**2,
                               (v_clipped - ret)**2).mean()

        entropy = entropies.mean()
        loss    = pi_loss + self.config['value_loss_coefficient'] * v_loss \
                  - beta * entropy

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.config['max_grad_norm'])
        self.optimizer.step()

        approx_kl = ((ratio - 1.0) - log_ratio).sum(dim=-1).mean()
        mean_log_ratio = log_ratio.sum(dim=-1).mean()

        return [pi_loss.item(), v_loss.item(), loss.item(), entropy.item(),
            approx_kl.item(), mean_log_ratio.item()]

    # ═══════════════════════════════════════════════════════════════
    # Checkpoint helpers
    # ═══════════════════════════════════════════════════════════════

    def _save_checkpoint(self, episode: int):
        path = os.path.join(self.ckpt_dir, f'ppo_{self.run_id}_ep{episode:06d}.pt')
        torch.save({
            'episode_count':  episode,
            'model':          self.model.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'update_count':   self.update_count,
        }, path)
        self.get_logger().info(f'Checkpoint saved → {path}')

    def _try_load_checkpoint(self):
        files = sorted([
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith(f'ppo_{self.run_id}') and f.endswith('.pt')
        ])
        if not files:
            return
        path = os.path.join(self.ckpt_dir, files[-1])
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.update_count = ckpt.get('update_count', 0)
        self.episode_count = ckpt.get('episode_count', 0)
        self.get_logger().info(f'Checkpoint loaded ← {path}')

    # ── Helpers ───────────────────────────────────────────────────

    def _cell_arrays(self):
        """Return (hx, cx) numpy arrays for buffer storage."""
        if not self.use_rec or self.recurrent_cell is None:
            return None, None
        if isinstance(self.recurrent_cell, tuple):
            hx, cx = self.recurrent_cell
            return hx.squeeze().cpu(), cx.squeeze().cpu()
        return self.recurrent_cell.squeeze().cpu(), None


# ─────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = PPOAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.writer.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
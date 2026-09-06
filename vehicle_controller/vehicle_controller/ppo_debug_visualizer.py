#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import csv
import os
import numpy as np
from collections import deque
from datetime import datetime
from std_msgs.msg import Float64MultiArray, String

import matplotlib.pyplot as plt


class PPODebugVisualizer(Node):

    def __init__(self):
        super().__init__('ppo_debug_visualizer')

        self.declare_parameter('history_size',  250)
        self.declare_parameter('checkpoint_dir', os.path.join(os.path.expanduser('~'), 'ppo_checkpoints'))
        self.declare_parameter('run_id',         'default')

        self.history_size = int(
            self.get_parameter('history_size').get_parameter_value().integer_value
        )
        checkpoint_dir = self.get_parameter('checkpoint_dir').get_parameter_value().string_value
        run_id         = self.get_parameter('run_id').get_parameter_value().string_value

        # -- Episode Queues --
        self.ep_x      = deque(maxlen=self.history_size)
        self.ep_reward = deque(maxlen=self.history_size)
        self.ep_length = deque(maxlen=self.history_size)
        self._trajectory_id = ''

        # -- Training Queues --
        self.up_x        = deque(maxlen=self.history_size)
        self.policy_loss = deque(maxlen=self.history_size)
        self.value_loss  = deque(maxlen=self.history_size)
        self.total_loss  = deque(maxlen=self.history_size)
        self.entropy     = deque(maxlen=self.history_size)
        self.lr          = deque(maxlen=self.history_size)
        self.update_ms   = deque(maxlen=self.history_size)

        # -- Reward Components Queues --
        self.reward_step     = deque(maxlen=self.history_size)
        self.reward_total    = deque(maxlen=self.history_size)
        self.reward_lat      = deque(maxlen=self.history_size)
        self.reward_lon      = deque(maxlen=self.history_size)
        self.reward_yaw      = deque(maxlen=self.history_size)
        self.reward_v        = deque(maxlen=self.history_size)
        self.reward_slew     = deque(maxlen=self.history_size)
        self.reward_rates    = deque(maxlen=self.history_size)
        self.reward_vz       = deque(maxlen=self.history_size)
        self.reward_res      = deque(maxlen=self.history_size)
        self.reward_progress = deque(maxlen=self.history_size)
        self.reward_term     = deque(maxlen=self.history_size)

        # ================================================================
        # CSV LOGGING SETUP
        # ================================================================
        log_dir = os.path.join(checkpoint_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        metrics_path = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        rewards_path = os.path.join(log_dir, f'rewards_{timestamp}.csv')
        ep_met_path  = os.path.join(log_dir, f'episodes_raw_{timestamp}.csv')

        self._metrics_file = open(metrics_path, 'w', newline='')
        self._rewards_file = open(rewards_path, 'w', newline='')
        self._ep_met_file  = open(ep_met_path,  'w', newline='')

        self._metrics_writer = csv.writer(self._metrics_file)
        self._rewards_writer = csv.writer(self._rewards_file)
        self._ep_met_writer  = csv.writer(self._ep_met_file)

        # Escribir encabezados
        self._metrics_writer.writerow([
              'total_env_steps', 'policy_loss', 'value_loss', 'total_loss', 'entropy', 'lr', 'update_ms', 'current_episode', 'approx_kl', 'mean_log_ratio'
        ])
        self._rewards_writer.writerow([
            'step', 'total', 'lat', 'lon', 'yaw', 'v', 'slew', 'rates', 'vz', 'res', 'progress', 'term'
        ])
        self._ep_met_writer.writerow([
            'episode', 'trajectory_id', 'raw_reward', 'raw_length_steps', 'at_update_step', 'total_env_steps'
        ])

        # Flush inmediato para que el encabezado quede escrito
        self._metrics_file.flush()
        self._rewards_file.flush()
        self._ep_met_file.flush()

        self.get_logger().info(f'Logging metrics  → {log_dir}')

        # ================================================================
        # SUBSCRIPTIONS & TIMER
        # ================================================================
        self.create_subscription(Float64MultiArray, '/ppo/episode_metrics',    self.ep_met_cb, 10)
        self.create_subscription(Float64MultiArray, '/ppo/train_metrics',  self.train_met_cb, 10)
        self.create_subscription(Float64MultiArray, '/ppo/reward_terms',   self.reward_cb, 10)
        self.create_subscription(String, '/rl/trajectory_id', self.trajectory_id_cb, 1)

        self.create_timer(0.1, self.update_plot)

        # ================================================================
        # 1. NN METRICS FIGURE (4x2)
        # ================================================================
        self.fig_metrics, self.axes_m = plt.subplots(4, 2, figsize=(7, 7), sharex=False)
        self.axes_m = self.axes_m.flatten()
        self.lines_m = {
            'reward':      self.axes_m[0].plot([], [], label='Reward',     color='darkblue')[0],
            'length':      self.axes_m[1].plot([], [], label='Length',     color='teal')[0],
            'policy_loss': self.axes_m[2].plot([], [], label='Policy Loss',color='crimson')[0],
            'value_loss':  self.axes_m[3].plot([], [], label='Value Loss', color='darkorange')[0],
            'total_loss':  self.axes_m[4].plot([], [], label='Total Loss', color='black')[0],
            'entropy':     self.axes_m[5].plot([], [], label='Entropy',    color='purple')[0],
            'lr':          self.axes_m[6].plot([], [], label='LR',         color='forestgreen')[0],
            'update_ms':   self.axes_m[7].plot([], [], label='Time (ms)',  color='gray')[0],
        }

        self.axes_m[0].set_title('Episode Reward')
        self.axes_m[1].set_title('Episode Length')
        self.axes_m[2].set_title('Policy Loss')
        self.axes_m[3].set_title('Value Loss')
        self.axes_m[4].set_title('Total Loss')
        self.axes_m[5].set_title('Entropy')
        self.axes_m[6].set_title('Learning Rate')
        self.axes_m[7].set_title('Update Time')

        self.axes_m[6].set_yscale('log')
        self.axes_m[6].set_ylim(1e-6, 1e-3)

        self.axes_m[0].set_xlabel('Episode')
        self.axes_m[1].set_xlabel('Episode')
        self.axes_m[6].set_xlabel('Env Steps')
        self.axes_m[7].set_xlabel('Env Steps')

        for ax in self.axes_m:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

        self.fig_metrics.tight_layout(h_pad=0.5)

        # ================================================================
        # 2. REWARD SIGNALS FIGURE (5x2)
        # ================================================================
        self.fig_rewards, self.axes_r = plt.subplots(5, 2, figsize=(7, 9), sharex=True)
        self.axes_r = self.axes_r.flatten()

        self.lines_r = {
            'lat':      self.axes_r[0].plot([], [], label='Lat',           color='red',       linewidth=1.5)[0],
            'lon':      self.axes_r[0].plot([], [], label='Lon',           color='blue',      alpha=0.7)[0],
            'yaw':      self.axes_r[1].plot([], [], label='Yaw',           color='purple')[0],
            'v':        self.axes_r[2].plot([], [], label='Velocity',      color='darkorange')[0],
            'slew':     self.axes_r[3].plot([], [], label='Slew Rate',     color='teal',      linewidth=1.8)[0],
            'rates':    self.axes_r[4].plot([], [], label='Angular Rates', color='darkgreen', linewidth=1.8)[0],
            'vz':       self.axes_r[5].plot([], [], label='Z-Vel',         color='firebrick', linewidth=1.8)[0],
            'res':      self.axes_r[6].plot([], [], label='Residual Pen',  color='steelblue', linewidth=1.8)[0],
            'progress': self.axes_r[7].plot([], [], label='Progress',      color='limegreen', linewidth=1.8)[0],
            'term':     self.axes_r[8].plot([], [], label='Terminal',      color='magenta',   linestyle='--')[0],
            'total':    self.axes_r[9].plot([], [], label='Total',         color='black',     linewidth=2.5)[0],
        }

        self.axes_r[0].set_title('Lat & Lon Error')
        self.axes_r[1].set_title('Yaw Error')
        self.axes_r[2].set_title('Velocity Penalty')
        self.axes_r[3].set_title('Slew Rate Penalty')
        self.axes_r[4].set_title('Angular Penalty')
        self.axes_r[5].set_title('Z-Vel Penalty')
        self.axes_r[6].set_title('Residual Penalty')
        self.axes_r[7].set_title('Track Progress')
        self.axes_r[8].set_title('Terminal Event')
        self.axes_r[9].set_title('Total Reward')

        for ax in self.axes_r:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

        self.axes_r[8].set_xlabel('Steps')
        self.axes_r[9].set_xlabel('Steps')

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.get_logger().info('PPO Debug Visualizer Initialized.')

    # ====================================================================
    # CALLBACKS
    # ====================================================================

    def trajectory_id_cb(self, msg: String):
        self._trajectory_id = msg.data

    def ep_met_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 5: return

        episode_num  = int(msg.data[0])
        raw_reward   = float(msg.data[1])
        raw_length   = int(msg.data[2])
        at_update    = int(msg.data[3])
        total_steps  = int(msg.data[4])

        self.ep_x.append(episode_num)
        self.ep_reward.append(raw_reward)
        self.ep_length.append(raw_length)

        self._ep_met_writer.writerow([
            episode_num, self._trajectory_id, raw_reward, raw_length, at_update, total_steps
        ])
        self._ep_met_file.flush()


    def train_met_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 8: return
        step = msg.data[0]
        self.up_x.append(step)
        self.policy_loss.append(msg.data[1])
        self.value_loss.append(msg.data[2])
        self.total_loss.append(msg.data[3])
        self.entropy.append(msg.data[4])
        self.lr.append(msg.data[5])
        self.update_ms.append(msg.data[6])

        approx_kl = msg.data[8] if len(msg.data) > 8 else ''
        mean_log_ratio = msg.data[9] if len(msg.data) > 9 else ''
        self._metrics_writer.writerow([
            step, msg.data[1], msg.data[2], msg.data[3], msg.data[4], msg.data[5],
            msg.data[6], msg.data[7], approx_kl, mean_log_ratio
        ])
        self._metrics_file.flush()

    def reward_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 12:
            return

        self.reward_step.append(msg.data[0])
        self.reward_total.append(msg.data[1])
        self.reward_lat.append(msg.data[2])
        self.reward_lon.append(msg.data[3])
        self.reward_yaw.append(msg.data[4])
        self.reward_v.append(msg.data[5])
        self.reward_slew.append(msg.data[6])
        self.reward_rates.append(msg.data[7])
        self.reward_vz.append(msg.data[8])
        self.reward_res.append(msg.data[9])
        self.reward_progress.append(msg.data[10])
        self.reward_term.append(msg.data[11])

        # --- Guardar en CSV ---
        self._rewards_writer.writerow([
            msg.data[0],  # step
            msg.data[1],  # total
            msg.data[2],  # lat
            msg.data[3],  # lon
            msg.data[4],  # yaw
            msg.data[5],  # v
            msg.data[6],  # slew
            msg.data[7],  # rates
            msg.data[8],  # vz
            msg.data[9],  # res
            msg.data[10], # progress
            msg.data[11], # term
        ])
        self._rewards_file.flush()   # escribe al disco de inmediato

    # ====================================================================
    # PLOT UPDATE
    # ====================================================================

    def update_plot(self):
        if len(self.ep_x) >= 1:
            ex = np.array(self.ep_x)
            self.lines_m['reward'].set_data(ex, np.array(self.ep_reward))
            self.lines_m['length'].set_data(ex, np.array(self.ep_length))
            self.axes_m[0].relim(); self.axes_m[0].autoscale_view()
            self.axes_m[1].relim(); self.axes_m[1].autoscale_view()

        if len(self.up_x) >= 1:
            ux = np.array(self.up_x)
            self.lines_m['policy_loss'].set_data(ux, np.array(self.policy_loss))
            self.lines_m['value_loss'].set_data(ux,  np.array(self.value_loss))
            self.lines_m['total_loss'].set_data(ux,  np.array(self.total_loss))
            self.lines_m['entropy'].set_data(ux,     np.array(self.entropy))
            self.lines_m['lr'].set_data(ux,          np.array(self.lr))
            self.lines_m['update_ms'].set_data(ux,   np.array(self.update_ms))
            
            for i in range(2, 8):
                self.axes_m[i].relim(); self.axes_m[i].autoscale_view()

        self.fig_metrics.canvas.draw_idle()
        self.fig_metrics.canvas.flush_events()

        # -- 2. Update Reward Plots --
        if len(self.reward_step) >= 2:
            rx = np.array(self.reward_step)
            self.lines_r['lat'].set_data(rx,      np.array(self.reward_lat))
            self.lines_r['lon'].set_data(rx,      np.array(self.reward_lon))
            self.lines_r['yaw'].set_data(rx,      np.array(self.reward_yaw))
            self.lines_r['v'].set_data(rx,        np.array(self.reward_v))
            self.lines_r['slew'].set_data(rx,     np.array(self.reward_slew))
            self.lines_r['rates'].set_data(rx,    np.array(self.reward_rates))
            self.lines_r['vz'].set_data(rx,       np.array(self.reward_vz))
            self.lines_r['res'].set_data(rx,      np.array(self.reward_res))
            self.lines_r['progress'].set_data(rx, np.array(self.reward_progress))
            self.lines_r['term'].set_data(rx,     np.array(self.reward_term))
            self.lines_r['total'].set_data(rx,    np.array(self.reward_total))

            for ax in self.axes_r:
                ax.relim()
                ax.autoscale_view()
            self.fig_rewards.canvas.draw_idle()
            self.fig_rewards.canvas.flush_events()

    # ====================================================================
    # CLEANUP
    # ====================================================================

    def destroy_node(self):
        self.get_logger().info('Closing CSV log files.')
        self._metrics_file.close()
        self._rewards_file.close()
        self._ep_met_file.close()
        super().destroy_node()


# ========================================================================
# MAIN
# ========================================================================

def main():
    rclpy.init()
    node = PPODebugVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
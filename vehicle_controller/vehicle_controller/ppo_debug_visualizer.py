#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from collections import deque
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt

class PPODebugVisualizer(Node):

    def __init__(self):
        super().__init__('ppo_debug_visualizer')

        self.declare_parameter('history_size', 250)
        self.history_size = int(
            self.get_parameter('history_size').get_parameter_value().integer_value
        )

        # -- PPO / PyTorch Metrics Queues --
        self.t_update    = deque(maxlen=self.history_size)
        self.reward_mean = deque(maxlen=self.history_size)
        self.length_mean = deque(maxlen=self.history_size)
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

        self.create_subscription(Float64MultiArray, '/ppo/metrics', self.metrics_cb, 10)
        self.create_subscription(Float64MultiArray, '/ppo/reward_terms', self.reward_cb, 10)
        
        self.create_timer(0.1, self.update_plot)

        # ================================================================
        # 1. NN METRICS FIGURE (4x2)
        # ================================================================
        self.fig_metrics, self.axes_m = plt.subplots(4, 2, figsize=(7, 7), sharex=True)
        self.axes_m = self.axes_m.flatten()
        self.lines_m = {
            'reward_mean': self.axes_m[0].plot([], [], label='Reward', color='darkblue')[0],
            'length_mean': self.axes_m[1].plot([], [], label='Length', color='teal')[0],
            'policy_loss': self.axes_m[2].plot([], [], label='Policy Loss', color='crimson')[0],
            'value_loss':  self.axes_m[3].plot([], [], label='Value Loss', color='darkorange')[0],
            'total_loss':  self.axes_m[4].plot([], [], label='Total Loss', color='black')[0],
            'entropy':     self.axes_m[5].plot([], [], label='Entropy', color='purple')[0],
            'lr':          self.axes_m[6].plot([], [], label='LR', color='forestgreen')[0],
            'update_ms':   self.axes_m[7].plot([], [], label='Time (ms)', color='gray')[0],
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
        self.axes_m[6].set_xlabel('Updates')
        self.axes_m[7].set_xlabel('Updates')

        for ax in self.axes_m:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

        # ================================================================
        # 2. REWARD SIGNALS FIGURE (4x2) 
        # ================================================================
        # Changed to 4x2 since Lat & Lon are now sharing a plot
        self.fig_rewards, self.axes_r = plt.subplots(5, 2, figsize=(7, 9), sharex=True)
        self.axes_r = self.axes_r.flatten()

        self.lines_r = {
            'lat':      self.axes_r[0].plot([], [], label='Lat',          color='red',       linewidth=1.5)[0],
            'lon':      self.axes_r[0].plot([], [], label='Lon',          color='blue',      alpha=0.7)[0],
            'yaw':      self.axes_r[1].plot([], [], label='Yaw',          color='purple')[0],
            'v':        self.axes_r[2].plot([], [], label='Velocity',     color='darkorange')[0],
            'slew':     self.axes_r[3].plot([], [], label='Slew Rate',    color='teal',      linewidth=1.8)[0],
            'rates':    self.axes_r[4].plot([], [], label='Angular Rates',color='darkgreen', linewidth=1.8)[0],
            'vz':       self.axes_r[5].plot([], [], label='Z-Vel',        color='firebrick', linewidth=1.8)[0],
            'res':      self.axes_r[6].plot([], [], label='Residual Pen', color='steelblue', linewidth=1.8)[0],
            'progress': self.axes_r[7].plot([], [], label='Progress',     color='limegreen', linewidth=1.8)[0],
            'term':     self.axes_r[8].plot([], [], label='Terminal',     color='magenta',   linestyle='--')[0],
            'total':    self.axes_r[9].plot([], [], label='Total',        color='black',     linewidth=2.5)[0],
        }

        self.axes_r[0].set_title('Lat & Lon Error')
        self.axes_r[1].set_title('Yaw Error')
        self.axes_r[2].set_title('Velocity Penalty')
        self.axes_r[3].set_title('Slew Rate Penalty')
        self.axes_r[4].set_title('Angular Penalty')
        self.axes_r[5].set_title('Z-Vel Penalty')
        self.axes_r[6].set_title('Residual Penalty')   # ← nuevo
        self.axes_r[7].set_title('Track Progress')     # ← nuevo
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

    def metrics_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 9:
            return

        step = msg.data[0]
        self.t_update.append(step)
        self.reward_mean.append(msg.data[1])
        self.length_mean.append(msg.data[2])
        self.policy_loss.append(msg.data[3])
        self.value_loss.append(msg.data[4])
        self.total_loss.append(msg.data[5])
        self.entropy.append(msg.data[6])
        self.lr.append(msg.data[7])
        self.update_ms.append(msg.data[8])

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

    def update_plot(self):
        # -- 1. Update Metrics Plots --
        if len(self.t_update) >= 2:
            tx = np.array(self.t_update)
            self.lines_m['reward_mean'].set_data(tx, np.array(self.reward_mean))
            self.lines_m['length_mean'].set_data(tx, np.array(self.length_mean))
            self.lines_m['policy_loss'].set_data(tx, np.array(self.policy_loss))
            self.lines_m['value_loss'].set_data(tx, np.array(self.value_loss))
            self.lines_m['total_loss'].set_data(tx, np.array(self.total_loss))
            self.lines_m['entropy'].set_data(tx, np.array(self.entropy))
            self.lines_m['lr'].set_data(tx, np.array(self.lr))
            self.lines_m['update_ms'].set_data(tx, np.array(self.update_ms))

            for ax in self.axes_m:
                ax.relim()
                ax.autoscale_view()
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
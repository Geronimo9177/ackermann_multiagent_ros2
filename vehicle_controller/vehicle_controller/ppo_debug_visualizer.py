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

        self.declare_parameter('history_size', 600)
        self.history_size = int(
            self.get_parameter('history_size').get_parameter_value().integer_value
        )

        # ── Colas de Métricas de Entrenamiento (PPO / PyTorch) ──
        self.t_update = deque(maxlen=self.history_size)
        self.reward_mean = deque(maxlen=self.history_size)
        self.length_mean = deque(maxlen=self.history_size)
        self.policy_loss = deque(maxlen=self.history_size)
        self.value_loss  = deque(maxlen=self.history_size)
        self.total_loss  = deque(maxlen=self.history_size)
        self.entropy     = deque(maxlen=self.history_size)
        self.lr          = deque(maxlen=self.history_size)
        self.update_ms   = deque(maxlen=self.history_size)

        # ── Colas de Componentes de Recompensa (Física del Chasis) ──
        self.reward_step = deque(maxlen=self.history_size)
        self.reward_total = deque(maxlen=self.history_size)
        self.reward_lat  = deque(maxlen=self.history_size)
        self.reward_lon  = deque(maxlen=self.history_size)
        self.reward_yaw  = deque(maxlen=self.history_size)
        self.reward_v    = deque(maxlen=self.history_size)
        
        self.reward_slew  = deque(maxlen=self.history_size) # Castigo de Slew Rate
        self.reward_rates = deque(maxlen=self.history_size) # Dinámica Angular Gated
        self.reward_az    = deque(maxlen=self.history_size) # Impacto Vertical Z
        self.reward_term  = deque(maxlen=self.history_size) # Bono/Castigo Terminal

        self.create_subscription(Float64MultiArray, '/ppo/metrics', self.metrics_cb, 10)
        self.create_subscription(Float64MultiArray, '/ppo/reward_terms', self.reward_cb, 10)
        
        self.create_timer(0.1, self.update_plot)

        # ════════════════════════════════════════════════════════════════
        # 1. FIGURA DE MÉTRICAS DE RED NEURONAL (4x2)
        # ════════════════════════════════════════════════════════════════
        self.fig_metrics, self.axes_m = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        self.axes_m = self.axes_m.flatten()
        self.lines_m = {
            'reward_mean': self.axes_m[0].plot([], [], label='Recompensa Promedio', color='darkblue')[0],
            'length_mean': self.axes_m[1].plot([], [], label='Pasos Promedio', color='teal')[0],
            'policy_loss': self.axes_m[2].plot([], [], label='Pérdida Política (Actor)', color='crimson')[0],
            'value_loss':  self.axes_m[3].plot([], [], label='Pérdida Valor (Crítico)', color='darkorange')[0],
            'total_loss':  self.axes_m[4].plot([], [], label='Pérdida Total', color='black')[0],
            'entropy':     self.axes_m[5].plot([], [], label='Entropía (Exploración)', color='purple')[0],
            'lr':          self.axes_m[6].plot([], [], label='Tasa de Aprendizaje', color='forestgreen')[0],
            'update_ms':   self.axes_m[7].plot([], [], label='Tiempo de Actualización', color='gray')[0],
        }

        self.axes_m[0].set_title('Rendimiento: Recompensa por Episodio')
        self.axes_m[1].set_title('Supervivencia: Pasos por Episodio')
        self.axes_m[2].set_title('Red Neuronal: Pérdida de Política')
        self.axes_m[3].set_title('Red Neuronal: Pérdida de Valor')
        self.axes_m[4].set_title('Red Neuronal: Pérdida Combinada')
        self.axes_m[5].set_title('Exploración: Entropía')
        self.axes_m[6].set_title('Optimizador: Tasa de Aprendizaje (LR)')
        self.axes_m[7].set_title('Rendimiento del Hardware: ms por Update')

        self.axes_m[6].set_yscale('log')
        self.axes_m[6].set_xlabel('Actualizaciones (Updates)')
        self.axes_m[7].set_xlabel('Actualizaciones (Updates)')

        for ax in self.axes_m:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

        # ════════════════════════════════════════════════════════════════
        # 2. FIGURA DE SEÑALES DE RECOMPENSA (3x3)
        # ════════════════════════════════════════════════════════════════
        self.fig_rewards, self.axes_r = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
        self.axes_r = self.axes_r.flatten()
        self.lines_r = {
            'lat':   self.axes_r[0].plot([], [], label='Castigo Lateral', color='red', linewidth=1.5)[0],
            'lon':   self.axes_r[1].plot([], [], label='Castigo Longitudinal', color='blue', alpha=0.7)[0],
            'yaw':   self.axes_r[2].plot([], [], label='Castigo Yaw', color='purple')[0],
            
            'v':     self.axes_r[3].plot([], [], label='Castigo Velocidad', color='darkorange')[0],
            'slew':  self.axes_r[4].plot([], [], label='Castigo Slew Rate (Δu)', color='teal', linewidth=1.8)[0],
            'rates': self.axes_r[5].plot([], [], label='Castigo Angular (Gated)', color='darkgreen', linewidth=1.8)[0],
            
            'az':    self.axes_r[6].plot([], [], label='Castigo Acel Vertical Z', color='firebrick', linewidth=1.8)[0],
            'term':  self.axes_r[7].plot([], [], label='Evento Terminal', color='magenta', linestyle='--')[0],
            'total': self.axes_r[8].plot([], [], label='RECOMPENSA TOTAL', color='black', linewidth=2.5)[0],
        }

        self.axes_r[0].set_title('1. Desviación Lateral')
        self.axes_r[1].set_title('2. Desfase Longitudinal')
        self.axes_r[2].set_title('3. Error de Orientación (Yaw)')
        
        self.axes_r[3].set_title('4. Exceso de Velocidad Nominal')
        self.axes_r[4].set_title('5. Slew Rate (Brusquedad de Acción)')
        self.axes_r[5].set_title('6. Dinámica Angular (Roll/Pitch Gated)')
        
        self.axes_r[6].set_title('7. Impacto Estructural Vertical (a_z)')
        self.axes_r[7].set_title('8. Evento Terminal (Crash/Success)')
        self.axes_r[8].set_title('9. SUMA DE RECOMPENSA (Total PPO)')

        for ax in self.axes_r:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')
            
        for ax in self.axes_r[6:]:
            ax.set_xlabel('Pasos de Simulación')

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.get_logger().info('Visualizador de PPO y Recompensas Inicializado.')

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
        # El PPO envía: [step, total, lat, lon, yaw, v, slew, rates_gated, az, terminal]
        if len(msg.data) < 10:
            return

        self.reward_step.append(msg.data[0])
        self.reward_total.append(msg.data[1])
        self.reward_lat.append(msg.data[2])
        self.reward_lon.append(msg.data[3])
        self.reward_yaw.append(msg.data[4])
        self.reward_v.append(msg.data[5])
        
        self.reward_slew.append(msg.data[6])
        self.reward_rates.append(msg.data[7])
        self.reward_az.append(msg.data[8])
        self.reward_term.append(msg.data[9])

    def update_plot(self):
        # ── 1. Actualizar Gráficos de Red Neuronal ──
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

        # ── 2. Actualizar Gráficos de Recompensa (Física) ──
        if len(self.reward_step) >= 2:
            rx = np.array(self.reward_step)
            self.lines_r['lat'].set_data(rx, np.array(self.reward_lat))
            self.lines_r['lon'].set_data(rx, np.array(self.reward_lon))
            self.lines_r['yaw'].set_data(rx, np.array(self.reward_yaw))
            self.lines_r['v'].set_data(rx, np.array(self.reward_v))
            self.lines_r['slew'].set_data(rx, np.array(self.reward_slew))
            self.lines_r['rates'].set_data(rx, np.array(self.reward_rates))
            self.lines_r['az'].set_data(rx, np.array(self.reward_az))
            self.lines_r['term'].set_data(rx, np.array(self.reward_term))
            self.lines_r['total'].set_data(rx, np.array(self.reward_total))

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
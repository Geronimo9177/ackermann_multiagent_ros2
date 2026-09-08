#!/usr/bin/env python3
"""
SpeedBumpSupervisor Controller
Controls the visibility of the 21 speed bumps via ROS2.
Runs as the supervisor in Webots.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from controller import Supervisor
import os
import random


class SpeedBumpSupervisor(Node):
    NUM_SPEEDBUMPS = 21
    NUM_HIDDEN = 4
    HIDDEN_Z = -20.0
    VISIBLE_POS = [0.0, 0.0, -2.45]
    
    def __init__(self):
        super().__init__('speedbump_supervisor')
        
        # Initialize the Webots supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.seed = int(os.environ.get('WEBOTS_SEED', '0'))
        random.seed(self.seed)
        
        # Store the translation fields of the speed bumps.
        self.speedbumps = {}
        self.currently_hidden = set()
        
        # Load references to the 21 speed bumps
        self._load_speedbumps()
        
        self.srv = self.create_service(
            Trigger,
            '/speedbump_control/reset_episode',
            self._reset_episode_cb
        )
        
        self.get_logger().info('SpeedBumpSupervisor started - waiting for commands...')
    
    def _load_speedbumps(self):
        """Load references to the 21 speed bumps and save their original positions."""
        for i in range(1, self.NUM_SPEEDBUMPS + 1):
            name = f'speedbump_{i:02d}'
            node = self.supervisor.getFromDef(name)
            
            if node is None:
                self.get_logger().error(f'{name} was not found in the world')
                continue
            
            translation_field = node.getField('translation')
            if translation_field is None:
                self.get_logger().error(f'{name} has no translation field')
                continue

            self.speedbumps[i] = translation_field
    
    def _reset_episode_cb(self, _request, response):
        available_indices = list(self.speedbumps)
        if len(available_indices) < self.NUM_HIDDEN:
            response.success = False
            response.message = (
                f'Only {len(available_indices)} speed bumps were found; '
                f'{self.NUM_HIDDEN} are required.'
            )
            return response

        selectable_indices = [idx for idx in available_indices if idx >= 2]
        if len(selectable_indices) < self.NUM_HIDDEN:
            response.success = False
            response.message = (
                f'Only {len(selectable_indices)} selectable speed bumps were found; '
                f'{self.NUM_HIDDEN} are required.'
            )
            return response

        new_hidden = set(random.sample(selectable_indices, self.NUM_HIDDEN))
        self.get_logger().info(
            f'Episode reset - removing speed bumps: {sorted(new_hidden)}')

        for idx in self.currently_hidden - new_hidden:
            self._set_z(idx, self.VISIBLE_POS[2])
        for idx in new_hidden - self.currently_hidden:
            self._set_z(idx, self.HIDDEN_Z)

        self.currently_hidden = new_hidden
        response.success = True
        response.message = f'Hidden: {sorted(new_hidden)}'
        return response

    def _set_z(self, idx, z):
        field = self.speedbumps.get(idx)
        if field is not None:
            field.setSFVec3f([self.VISIBLE_POS[0], self.VISIBLE_POS[1], z])
    
    def run(self):
        """Main loop that integrates Webots with ROS2."""
        while self.supervisor.step(self.timestep) != -1:
            # Process ROS2 callbacks
            rclpy.spin_once(self, timeout_sec=0.001)
        
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    supervisor = SpeedBumpSupervisor()
    supervisor.run()


if __name__ == '__main__':
    main()

from glob import glob
from setuptools import find_packages, setup

package_name = 'vehicle_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/trajectories', glob('trajectories/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='geronimo',
    maintainer_email='geronimohur@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'ackermann_mpc = vehicle_controller.ackermann_mpc:main',
        'topp = vehicle_controller.topp:main',
        'trajectory_publisher = vehicle_controller.trayectory_publisher:main',
        'mpc_debug_visualizer = vehicle_controller.mpc_debug_visualizer:main',
    ],
},

)
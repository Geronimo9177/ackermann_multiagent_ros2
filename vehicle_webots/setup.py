from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vehicle_webots'


def install_tree(folder):
    return [
        (
            os.path.join('share', package_name, root),
            [os.path.join(root, f) for f in files],
        )
        for root, _, files in os.walk(folder)
        if files
    ]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.wbt')),
        ('share/' + package_name + '/resource', glob('resource/*.urdf')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ] + install_tree('models'),
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
            "car_driver = vehicle_webots.car_driver:main",
        ],
    },
)

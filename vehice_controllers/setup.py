from setuptools import find_packages, setup

package_name = 'vehice_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'odom_comparer = vehice_controllers.odom_comparer:main',
            'joint_controller_plotter = vehice_controllers.joint_controller_plotter:main',
            'steering_controller_plotter = vehice_controllers.steering_controller_plotter:main',
        ],
    },
)

import os
from glob import glob
from setuptools import setup

package_name = 'rdk_bottle_hunter'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'model'), glob('model/*.bin')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rdk',
    maintainer_email='dev@example.com',
    description='ROS2 bottle hunter: YOLOv5 detection, LiDAR avoidance, STM32 motor control.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_detector_node = rdk_bottle_hunter.camera_detector_node:main',
            'motor_driver_node = rdk_bottle_hunter.motor_driver_node:main',
            'controller_node = rdk_bottle_hunter.controller_node:main',
        ],
    },
)

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'maintenance_ai'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teja',
    maintainer_email='teja@todo.todo',
    description='AI health classification and anomaly detection for the maintenance robot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'health_node = maintenance_ai.ros_node:main',
            'train_model = maintenance_ai.train:main',
            'replay = maintenance_ai.replay:main',
        ],
    },
)

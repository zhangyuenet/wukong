from glob import glob

from setuptools import find_packages, setup

package_name = 'wukong_flight'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yue Zhang',
    maintainer_email='noreply@example.com',
    description='Flight control layer for Wukong.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fly_mission = wukong_flight.fly_mission_node:main',
        ],
    },
)

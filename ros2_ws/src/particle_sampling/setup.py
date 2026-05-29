from setuptools import find_packages, setup

package_name = 'particle_sampling'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Navid',
    maintainer_email='navid@example.com',
    description='ROS 2 node that samples particles in front of the robot from a CSV trajectory file',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_sampling_node = particle_sampling.particle_sampling_node:main',
        ],
    },
)

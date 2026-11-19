from setuptools import setup

package_name = 'hspar_viz'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Navid',
    maintainer_email='navid@example.com',
    description='Illustration-only ROS 2 vector field publisher for RViz',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hspar-viz = hspar_viz.hspar_viz_node:main',
        ],
    },
)
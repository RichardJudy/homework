from setuptools import setup, find_packages

package_name = 'color_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),  # 解决marker警告
        ('share/' + package_name, ['package.xml']),  # 解决package.xml警告
        ('share/' + package_name + '/launch', ['launch/color_detection.launch.py']),
    ],
    install_requires=['setuptools', 'opencv-python'],
    zip_safe=True,
    maintainer='zyy',
    maintainer_email='zyy@example.com',
    description='ROS2 color detection package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 正确注册摄像头节点
            'camera_node = color_detection.camera_node:main',
            # 正确注册颜色检测节点
            'color_detect_node = color_detection.color_detect_node:main',
            # 如果不需要color_node，可删除此行；如果需要保留，确保对应文件存在
            # 'color_node = color_detection.color_node:main',
        ],
    },
)


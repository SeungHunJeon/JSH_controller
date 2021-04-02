from setuptools import setup

package_name = 'rqt_raisin_lc'
setup(
    name=package_name,
    version='1.0.0',
    package_dir={'': 'src'},
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/resource', ['resource/LearningControl.ui']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['plugin.xml']),
        ('lib/' + package_name, ['scripts/rqt_raisin_lc'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Donghoon Youm',
    maintainer='Donghoon Youm',
    maintainer_email='ydh0725@kaist.ac.kr',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'rqt_raisin_lc provides a GUI plugin to  control a robot using Command srv'
    ),
    license='BSD',
    tests_require=['pytest'],
    scripts=['scripts/rqt_raisin_lc'],
)

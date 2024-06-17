# setup.py in the LightSailSim root directory

from setuptools import setup, find_packages

setup(
    name='LightSailSim',
    version='0.1',
    packages=find_packages(include=['src', 'src.*']),
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
    ],
)

from setuptools import setup, find_packages

setup(
    name='grasp_gym',
    version='0.1',
    packages=find_packages(),  # Find packages automatically within the current directory
    install_requires=['gymnasium']
)
from setuptools import setup, find_packages

setup(
    name="e11scan",
    version="0.1.2",
    author="Luke Brown",
    packages=find_packages(exclude=['*tests']),
    install_requires=['h5py', 'pandas', 'scipy', 'numpy', 'colorama', 'matplotlib']
)

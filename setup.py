import os
from setuptools import setup, find_packages
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


setup(
    name='vrag',
    version='0.0.2',
    url='https://github.com/Kennard123661/VRAG',
    packages=find_packages(),
    license='MIT License',
    author='kennardng',
    author_email='e0036319@u.nus.edu',
    description='vrag',
    install_requires=[
        'h5py', 'tqdm', 'scikit-video', 'opencv-python',
        'tensorboard', 'tensorboardX', 'tqdm', 'scikit-learn',
        'torch', 'torchvision'
    ]
)



import os
from setuptools import setup, find_packages
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


def read_requirements() -> list:
    requirement_fp = os.path.join(PROJECT_DIR, 'requirements.txt')
    with open(requirement_fp, 'r') as f:
        requirements = f.readlines()
    requirements = [requirement.strip() for requirement in requirements]
    return requirements


setup(
    name='vrag',
    version='',
    url='https://github.com/Kennard123661/VRAG',
    packages=find_packages(),
    license='MIT License',
    author='kennardng',
    author_email='e0036319@u.nus.edu',
    description='Video Region Attention Graph for Content-based Video Retrieval',
    install_requires=read_requirements()
)



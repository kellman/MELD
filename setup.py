import sys
from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['numpy', 'scipy', 'sigpy', 'torch']

setup(
    name='MELD',
    version='0.1dev',
    packages=['meld',],
    author='Michael Kellman',
    author_email='kellman@berkeley.edu',
    license='BSD',
    long_description=open('README.md').read(),
)
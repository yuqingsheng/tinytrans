from setuptools import setup, find_packages
import sys, os
import re
import time

version = '1.0.0'

if not version:
    raise RuntimeError('Cannot find version information')

with open('README.md', 'rb') as f:
    readme = f.read().decode('utf-8')

install_requires = []
with open('requirements.txt', 'r') as f:
    for req in f.readlines():
        install_requires.append(req.strip())


setup(
    name='tinytrans',
    version=version,
    description="Feature Smart Transform",
    long_description=readme,
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
    author='yuqingsheng',
    author_email='qingsheng.yu@ishansong.com',
    url='http://www.yongqianbao.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)

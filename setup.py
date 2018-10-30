#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from distutils.core import setup

with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()


setup(name='qcopt',
      version='1.0.0',
      author='Kazuhiro HISHINUMA, Hideaki IIDUKA',
      author_email='kaz@cs.meiji.ac.jp',
      description='An implementation of the fixed point quasiconvex subgradient method',
      long_description=LONG_DESCRIPTION,
      url='https://iiduka.net',
      packages=['qcopt'],
      install_requires=[
          'numpy>=1.15.0',
          'scipy>=1.1.0',
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: CPython'
          'License :: OSI Approved :: MIT License',
      ]
)

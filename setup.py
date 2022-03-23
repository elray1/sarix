#!/usr/bin/env python

from distutils.core import setup

setup(name='sarix',
    version='0.0.1',
    description='Seasonal AR, Integrated models with eXogenous predictors',
    author='Evan L. Ray',
    author_email='elray@umass.edu',
    url='https://github.com/elray1/sarix',
    packages=['sarix'],
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'numpyro',
        'jax',
        'covidcast'
    ],
)

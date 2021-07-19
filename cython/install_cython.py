""""    
Author: Bjarne Johannsen
Python Version: 3.8

Cython Setup.
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('cython.pyx'))
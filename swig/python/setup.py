#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


example_module = Extension('_tinyclassifier',
                           sources=['tinyclassifier_wrap.cxx'],
                           )

setup (name = 'tinyclassifier',
       version = '0.1',
       description = """python extension of tinyclassifier""",
       py_modules = ['tinyclassifier'],
       ext_modules = [Extension('_tinyclassifier',
                                ["tinyclassifier_wrap.cxx"],
                                include_dirs=["../../include"],
                                library_dirs=["../../lib"],
                                libraries=["tinyclassifier","gcc","stdc++"])]
       )

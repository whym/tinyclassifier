#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

example_module = Extension('_TinyClassifier',
                           sources=['TinyClassifier_wrap.cxx'],
                           )

setup (name = 'TinyClassifier',
       version = '0.1',
       description = """python extension of tinyclassifier""",
       py_modules = ['TinyClassifier'],
       ext_modules = [Extension('_TinyClassifier',
                                ['../TinyClassifier.i', 'TinyClassifier_wrap.cxx'],
                                include_dirs=['../../include'],
                                library_dirs=[],
                                libraries=['gcc','stdc++'])]
       )

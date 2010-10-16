#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import os
import sys

conf = {'CXXFLAGS' : [], 'LDFLAGS' : []}
for x in conf.keys():
    if os.environ.has_key(x):
        conf[x] += filter(lambda x: x != '', os.environ[x].split(' '))

setup (name = 'TinyClassifier',
       version = '0.1',
       description = """python extension of tinyclassifier""",
       py_modules = ['TinyClassifier'],
       ext_modules = [Extension('_TinyClassifier',
                                ['TinyClassifier_wrap.cxx'],
                                swig_opts=['-c++', '-modern', '-I../../include', '-Wall'],
                                include_dirs = ['../../include'],
                                library_dirs = [],
                                #undef_macros=['NDEBUG'],
                                extra_compile_args = conf['CXXFLAGS'],
                                extra_link_args    = conf['LDFLAGS'],
                                libraries = ['gcc','stdc++'])]
       )

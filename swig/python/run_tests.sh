#! /bin/sh
env LD_LIBRARY_PATH="../../lib:${LD_LIBRARY_PATH}" PYTHONPATH=`python -c "import sys; from distutils.util import get_platform as p; print 'build/lib.%s-%d.%d' % ((p(),)+sys.version_info[0:2])"`:${PYTHONPATH} python test_perceptron.py

#! /bin/sh
env LD_LIBRARY_PATH="../../lib:${LD_LIBRARY_PATH}" PYTHONPATH=build/lib.linux-`uname -m`-`python --version 2>&1 |  sed 's/^[^ ]* \(2\.[0-9]*\).*/\1/'` python test_perceptron.py

#! /bin/sh
env LD_LIBRARY_PATH=../lib ./sample && echo yes && exit 0
echo no && exit 1

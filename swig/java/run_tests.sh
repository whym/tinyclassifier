#! /bin/sh
env LD_LIBRARY_PATH=".:../../lib:${LD_LIBRARY_PATH}" java -cp TinyClassifier.jar:$CLASSPATH org.whym.tinyclassifier.test.TestTinyClassifier

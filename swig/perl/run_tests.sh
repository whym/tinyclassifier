#! /bin/sh
env LD_LIBRARY_PATH="../../lib:${LD_LIBRARY_PATH}" PERL5LIB=blib/arch/auto/TinyClassifier:blib/lib:${PERL5LIB} perl test_perceptron.pl

CXXFLAGS := $(CXXFLAGS) -I../include -L../lib -ltinyclassifier -lstdc++
CC := $(CXX)

all: .succeed
.succeed: sample
	./run_tests.sh && touch $@
sample: sample.o ../lib/libtinyclassifier.so
	$(CXX) $(CXXFLAGS) -o $@ $<
%o: %.cc ../include/perceptron.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
clean:
	-rm  sample sample.o

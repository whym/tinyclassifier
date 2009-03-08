CC = $(CXX)
CXXFLAGS := $(CXXFLAGS) -fPIC -I../include -Wall
libtinyclassifier.so: perceptron.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $<
clean:
	-rm libtinyclassifier.so perceptron.o
perceptron.o: perceptron.cc ../include/util.h
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

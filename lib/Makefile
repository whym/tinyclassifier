CC = $(CXX)
CXXFLAGS := $(CXXFLAGS) -fPIC -I../include
libtinyclassifier.so: perceptron.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $<
clean:
	-rm libtinyclassifier.so perceptron.o
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

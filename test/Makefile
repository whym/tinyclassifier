CXXFLAGS := $(CXXFLAGS) -I../include -L../lib -ltinyclassifier -lstdc++
CC := $(CXX)

sample: sample.o
	$(CXX) $(CXXFLAGS) -o $@ $<
%o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
clean:
	-rm  sample sample.o

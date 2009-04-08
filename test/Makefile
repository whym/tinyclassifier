CXXFLAGS := $(CXXFLAGS) -I../include -L../lib -lstdc++ -Wall
CC := $(CXX)
SOURCES = sample.cc
HEADERS = ../include/*.h
DEPFILE = .depend.makefile

all: .succeed
.succeed: sample
	./run_tests.sh
sample: sample.o
	$(CXX) $(CXXFLAGS) -o $@ $<
%o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
clean:
	-rm  sample sample.o $(DEPFILE)

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

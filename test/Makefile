CXXFLAGS := $(CXXFLAGS) -I../include -L../lib -lstdc++ -Wall
CC := $(CXX)
RESULTS = sample
SOURCES = *.cc
HEADERS = ../include/*.h
DEPFILE = .depend.makefile

all: .succeed
.succeed: build
	./run_tests.sh
build: $(RESULTS)
clean:
	-rm  $(RESULTS) *.o $(DEPFILE)

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

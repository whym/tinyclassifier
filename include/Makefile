LDFLAGS ?= 
RESULTS = tester
SOURCES =
HEADERS = util.h lru_cache.h
DEPFILE = .depend.makefile

all: test
#all: test build
build: $(RESULTS)
clean:
	-rm $(RESULTS) *.o
	-rm *__
test:
	$(CXX) $(CXXFLAGS) -DTEST_TINYCLASSIFIER_UTIL_H tester.cc -o tester && ./tester
	$(CXX) $(CXXFLAGS) -DTEST_TINYCLASSIFIER_LRUCACHE_H tester.cc -o tester && ./tester
	$(CXX) $(CXXFLAGS) -DTEST_TINYCLASSIFIER_PERCEPTRON_H tester.cc -o tester && ./tester

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null
	cat $@

# NOTE: run make depend first (for resolving header file dependencies)
-include $(DEPFILE)

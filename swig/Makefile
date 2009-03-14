SWIG = swig
SWIGNAME = tinyclassifier
SOURCES = tinyclassifier.i
HEADERS = ../include/*.h
DEPFILE = .depend.makefile
CXXFLAGS = -I../include -Wall

all: depend build test
build: .ruby .perl .python

ruby/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig $(CXXFLAGS) -c++ -ruby -o $@ $<
.ruby: ruby/$(SWIGNAME)_wrap.cxx
	cd ruby && ruby extconf.rb && make

perl/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig $(CXXFLAGS) -c++ -perl -o $@ -shadow $<
.perl: perl/$(SWIGNAME)_wrap.cxx
	cd perl && make && make

python/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig $(CXXFLAGS) -c++ -python -o $@ -shadow $<
.python: python/$(SWIGNAME)_wrap.cxx
	cd python && python setup.py build

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

clean:
	-rm $(DEPFILE)
	make -C ruby   $@
	make -C perl   $@
	make -C python $@

test:
	cd ruby   && sh ./run_tests.sh
	cd perl   && sh ./run_tests.sh
	cd python && sh ./run_tests.sh

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

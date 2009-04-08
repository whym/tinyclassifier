SWIG = swig
SWIGNAME = TinyClassifier
SOURCES = TinyClassifier.i
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
	cd perl && perl Makefile.PL && make

python/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig $(CXXFLAGS) -c++ -python -o $@ -shadow $<
.python: python/$(SWIGNAME)_wrap.cxx
	cd python && python setup.py build

java/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig $(CXXFLAGS) -c++ -java -o $@ -package com.github.whym -shadow $<
	mkdir -p ./java/com/github/whym
	mv -f java/*.java ./java/com/github/whym
.java: java/$(SWIGNAME)_wrap.cxx
	cd java && make

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

clean:
	-rm $(DEPFILE) */$(SWIGNAME)_wrap.cxx
	make -C ruby   $@
	make -C perl   $@
	make -C python $@
	make -C java   $@
test:
	cd ruby   && sh ./run_tests.sh
	cd perl   && sh ./run_tests.sh
	cd python && sh ./run_tests.sh
	cd java   && sh ./run_tests.sh

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

SWIG = swig
SWIGNAME = TinyClassifier
SOURCES = TinyClassifier.i
HEADERS = ../include/*.h
DEPFILE = .depend.makefile
SWIGFLAGS = -I../include -Wall

all: depend build test
build: .ruby .perl .python

ruby/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -ruby -o $@ $<
.ruby: ruby/$(SWIGNAME)_wrap.cxx
	cd ruby && ruby extconf.rb && $(MAKE)

perl/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -perl -o $@ -shadow $<
.perl: perl/$(SWIGNAME)_wrap.cxx
	cd perl && perl Makefile.PL && $(MAKE) LD_RUN_PATH=""

python/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -python -o $@ -shadow $<
.python: python/$(SWIGNAME)_wrap.cxx
	cd python && python setup.py build

java/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -java -o $@ -package com.github.whym -shadow $<
	mkdir -p ./java/com/github/whym
	mv -f java/*.java ./java/com/github/whym
.java: java/$(SWIGNAME)_wrap.cxx
	cd java && $(MAKE)

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) $(SWIGFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

clean:
	-rm $(DEPFILE) */$(SWIGNAME)_wrap.cxx
	$(MAKE) -C ruby   $@
	$(MAKE) -C perl   $@
	$(MAKE) -C python $@
	$(MAKE) -C java   $@
test:
	cd ruby   && sh ./run_tests.sh
	cd perl   && sh ./run_tests.sh
	cd python && sh ./run_tests.sh
	$(MAKE) -C java $@

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

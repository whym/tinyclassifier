SWIG = swig
SWIGNAME = TinyClassifier
SOURCES = TinyClassifier.i
HEADERS = ../include/*.h
DEPFILE = .depend.makefile
SWIGFLAGS = -I../include -Wall

all: depend
	-$(MAKE) .ruby
	-$(MAKE) .perl
	-$(MAKE) .python
	-$(MAKE) .java

ruby/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS) depend
	$(SWIG) $(SWIGFLAGS) -c++ -ruby -o $@ $<
.ruby: ruby/$(SWIGNAME)_wrap.cxx
	cd ruby && ruby extconf.rb && $(MAKE) && sh ./run_tests.sh

perl/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -perl -o $@ -shadow $<
.perl: perl/$(SWIGNAME)_wrap.cxx
	cd perl && perl Makefile.PL && $(MAKE) LD_RUN_PATH="" && sh ./run_tests.sh

python/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -python -modern -o $@ -shadow $<
.python: python/$(SWIGNAME)_wrap.cxx
	cd python && python setup.py build && sh ./run_tests.sh

java/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	$(SWIG) $(SWIGFLAGS) -c++ -java -outdir ./java/com/github/whym/tinyclassifier -o $@ -package com.github.whym.tinyclassifier -shadow $<
.java: java/$(SWIGNAME)_wrap.cxx
	cd java && $(MAKE) && $(MAKE) test

depend: $(DEPFILE)
$(DEPFILE): $(SOURCES) $(HEADERS)
	makedepend -f- -- $(CXXFLAGS) $(SWIGFLAGS) -- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

clean:
	-rm $(DEPFILE) */$(SWIGNAME)_wrap.cxx
	-$(MAKE) -C ruby   $@
	-$(MAKE) -C perl   $@
	-$(MAKE) -C python $@
	-$(MAKE) -C java   $@
	cd python && python setup.py clean

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

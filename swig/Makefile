SWIG = swig
SWIGNAME = tinyclassifier
SOURCES = tinyclassifier.i
HEADERS = ../include/*.h
DEPFILE = .depend.makefile

all: depend .ruby

ruby/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig -I../include -Wall -c++ -ruby -o $@ $<
.ruby: ruby/$(SWIGNAME)_wrap.cxx
	cd ruby && ruby extconf.rb
	make -C ruby

perl/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig -I../include -Wall -c++ -perl -o $@ -shadow $<
.perl: perl/$(SWIGNAME)_wrap.cxx
	cd perl && perl Makefile.PL
	make -C perl

python/$(SWIGNAME)_wrap.cxx: $(SWIGNAME).i $(HEADERS)
	swig -I../include -Wall -c++ -python -o $@ -shadow $<
.python: python/$(SWIGNAME)_wrap.cxx
	cd python && python setup.py
	make -C python

depend: $(DEPFILE)
$(DEPFILE):
	makedepend -f- $(SOURCES) $(HEADERS) > $@ 2> /dev/null

# NOTE: run make depend first (for resolving header file dependencies)
include $(DEPFILE)

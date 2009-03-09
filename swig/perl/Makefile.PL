#! /usr/bin/env perl
use warnings;
use strict;
use ExtUtils::MakeMaker;

WriteMakefile(
              INC    => q{-I../../include},
              CC     => $ENV{'CXX'} || q{g++},
              LIBS   => q{-lstdc++ -L../../lib -ltinyclassifier},
              LINKTYPE=> q{dynamic},
              NAME   => q{tinyclassifier},
              OBJECT => q{tinyclassifier_wrap.o},
              CCFLAGS => q{-Wall -shared -fno-strict-aliasing},
              clean => {FILES => 'tinyclassifier_wrap.cxx'}
             );

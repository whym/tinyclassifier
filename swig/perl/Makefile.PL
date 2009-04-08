#! /usr/bin/env perl
use warnings;
use strict;
use ExtUtils::MakeMaker;
my $swigname = 'TinyClassifier';
WriteMakefile(
              INC    => q{-I../../include},
              CC     => $ENV{'CXX'} || q{g++},
#              LIBS   => q{-lstdc++ -L../../lib -ltinyclassifier},
              LIBS   => q{-lstdc++},
              LINKTYPE=> q{dynamic},
              NAME   => qq{$swigname},
              OBJECT => qq{${swigname}_wrap.o},
              CCFLAGS => q{-Wall -shared -fno-strict-aliasing},
              clean => {FILES => "${swigname}_wrap.cxx"}
             );

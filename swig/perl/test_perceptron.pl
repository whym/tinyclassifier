#! /usr/bin/env perl

use Test::More qw/no_plan/;
use TinyClassifier;
use Storable qw/freeze thaw/;
ok(TinyClassifier::power_int(11,2) == 11**2, 'power');
my $v = TinyClassifier::IntVector->new([11,22,33]);
ok(freeze([map($v->get($_), (0..2))]) eq freeze([11,22,33]), 'vector');
my $p = TinyClassifier::IntPerceptron->new(3);

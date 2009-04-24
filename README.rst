======================================================================
TinyClassifier - a tiny machine learning library for scripting
======================================================================


Overview
==============================

Tuning machine-learning based systems is an art.  You need to do
try-and-error again and again.  In such development cycles, it's
painful to spend the time for solving compilation errors.

Using scripting language is good for rapid programming.  But it's
not good for the CPU-intensive calculations which might be optimized
and/or parallelized by compilers.

TinyClassifier tries to fill the gap between a high-performance
inner implementation and application programs that use it.  With
TinyClassifier, you don't need to use pipes or files, in which you
have to encode/decode feature vectors via strings.  I believe the
elimination of such redundant work increases productivity.

TinyClassifier is a fast an flexible machine learning library with
following features:

- Small and self-contained implementation
- Reasonably efficient and compact implementation with C++
- Interfaces to Ruby, Perl, Python, etc. via SWIG

On the other hand, TinyClassifier is not for those people

- Who want the best accuracy and efficiency of machine learning.
- Who can productively implement anything in C++.


Implemented algorithms
==============================

Averaged Perceptron for binary classification
  non-kernelized version and kernelized version
  (currently polynomial kernel only)

Planned to implement
------------------------------

- Maximum entropy classifier with Stochastic Gradient Descent algorithm
- Complementary naive Bayes classifier


Sample usage
==============================

Ruby
------------------------------

::

  require 'TinyClassifier'
  include TinyClassifier

  SAMPLES = {
    [-2, +1, -1] => +1,
    [-1, +2, +1] => +1,
    [-1, -1, -1] => -1,
    [+1, +1, -1] => +1,
    [-1, +1, -1] => +1,
    [+1, -2, -1] => -1,
    [+1, -1, +1] => -1
  }

  keys = SAMPLES.keys.sort
  p = IntPKPerceptron.new(SAMPLES.keys[0].length, 10)
  p.train(IntVectorVector.new(keys),
          IntVector.new(keys.map{|x| SAMPLES[x]}))
  keys.each do |k|
    pred = p.predict(k)
    puts "#{SAMPLES[k]}: #{pred}"
  end


Requirements
==============================

- gcc 4.3 (possibly gcc 4.x)
- swig 1.3.35 (possibly swig 1.3.x)


Build & Install
==============================

C++ library
------------------------------

1. (Optional) Type 'make' at the top directory of tinyclassifier and
   the tests will run.

2. Copy the header files in 'include/' to an appropriate directory
   included in CPATH.

Recommended compile options
    For a multicore processor,
    ::
    
      CXXFLAGS="-ftree-vectorizer-verbose=1 -msse2 -ftree-vectorize -O3"

SWIG bindings
------------------------------

1. Type 'make -C swig' at the top directory of tinyclassifier.

2. (for Ruby) Go to the directory 'swig/ruby' and type 'make
      install'.
   
   (Alternatively) Copy 'swig/ruby/TinyClassifier.so' to somewhere
   included in RUBYLIB.

For some language bindings, you might have to manually install
library files.

# Details to be written


Usage
==============================

See the tests included in the package for the examples of usage.
Tests are located at 'test', 'swig/ruby/test', etc.


Homepage
==============================

http://whym.github.com/tinyclassifier


Notes
==============================

'PKPerceptron' is based on Ling-Pipe's explanation of Kernel Averaged
Perceptron. (see below for further information)

http://alias-i.com/lingpipe/docs/api/com/aliasi/classify/PerceptronClassifier.html



.. Local variables:
.. mode: rst
.. End:

=====================
TinyClassifier
=====================
------------------------------------------------
a tiny machine learning library for scripting
------------------------------------------------

 :Homepage: http://whym.github.com/tinyclassifier
 :Contact:  http://github.com/whym

Overview
==============================

Tuning machine-learning based systems is an art.  You need to do
try-and-error again and again.  In such development cycles, you might
want to avoid using such a language like C++; it can be painful to
spend the time for solving compilation errors.

Using scripting language is good for rapid programming.  But it's
not good for the CPU-intensive calculations, which should be optimized
and/or parallelized by compilers.

TinyClassifier tries to fill the gap between a high-performance inner
implementation and application programs that use it.  With
TinyClassifier, you don't need to use pipes or temporary files nor to
encode/decode feature vectors into/from strings.  Instead, you will
have transparent access to the data structures and class libraries
implemented in C++ from scripting languages like Python or Ruby.

TinyClassifier is a fast and flexible machine learning library that
provides you:

- **Small and self-contained** software package of machine learning
  with minimum dependency to external libraries
- Reasonably **efficient and readable** implementation as C++ header
  libraries
- **Language bindings** to Ruby, Perl, Python, etc. via SWIG

On the other hand, TinyClassifier is not for those people

- Who want the best accuracy and efficiency of machine learning.
- Who can productively implement anything in C++.

Implemented algorithms
==============================

Averaged Perceptron for binary classification
  
  Non-kernelized version and kernelized [#]_ version are implemented
  (currently polynomial kernel only).
  
.. [#]
  
  The implementation of 'PKPerceptron' is based on Ling-Pipe's
  explanation of Kernel Averaged Perceptron. See below for further
  information.
  
  http://alias-i.com/lingpipe/docs/api/com/aliasi/classify/PerceptronClassifier.html

Planned to implement
------------------------------

- Maximum entropy classifier with Stochastic Gradient Descent algorithm
- Complementary naive Bayes classifier

Requirements
==============================
Following softwares are required.

- gcc and g++ 4.3 (possibly gcc 4.x)
- swig 1.3.35 (possibly swig 1.3.x)
- make
- makedepend

The development environment needs to be prepared for each language you
will use with TinyClassifier. [#]_ Currently, language bindings are
maintained for the languages below.

- Python
- Perl 5
- Ruby
- Java

.. [#]
   
   For every language you wish to have the TinyClassifier library, you
   need to prepare development environment; normally you need to set
   up runtimes, compilers and API files appropriately.  Note that they
   sometimes are provided separately.  For example, you may need to
   install something like 'libruby' or 'sun-jdk-\*' to have API files
   installed.

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

Sample codes
==============================

Python ::
    
      from TinyClassifier import *
      
      # Prepare examples
      SAMPLES = [
          [[-2, +1, -1], +1],
          [[-1, +2, +1], +1],
          [[-1, -1, -1], -1],
          [[+1, +1, -1], +1],
          [[-1, +1, -1], +1],
          [[+1, -2, -1], -1],
          [[+1, -1, +1], -1]
          ]
      
      vecs = sorted([x[0] for x in SAMPLES]) # Obtain feature vectors
      labs = sorted([x[1] for x in SAMPLES]) # Obtain labels
      p = IntPKPerceptron(len(SAMPLES[0]), 10) # Construct a perceptron that stops after 10 iterations
      p.train(IntVectorVector(vecs),           # Give the perceptron training examples
              IntVector(labs))
      for (i, k) in enumerate(vecs):  # Print the prediction for the training examples (closed set evaluation)
          pred = p.predict(k)
          print "%d: %f" % (SAMPLES[i][1], pred)
    
Ruby ::
    
      require 'TinyClassifier'
      include TinyClassifier
      
      # Prepare examples
      SAMPLES = {
        [-2, +1, -1] => +1,
        [-1, +2, +1] => +1,
        [-1, -1, -1] => -1,
        [+1, +1, -1] => +1,
        [-1, +1, -1] => +1,
        [+1, -2, -1] => -1,
        [+1, -1, +1] => -1
      }
    
      keys = SAMPLES.keys.sort  # Obtain feature vectors
      keys = keys.map{|x| SAMPLES[x]} # Obtain labels
      p = IntPKPerceptron.new(SAMPLES.keys[0].length, 10) # Construct a perceptron that stops after 10 iterations
      p.train(IntVectorVector.new(keys),                  # Give the perceptron training examples
              IntVector.new(labels))
      keys.each do |k|          # Print the prediction for the training examples (closed set evaluation)
        pred = p.predict(k)
        puts "#{SAMPLES[k]}: #{pred}"
      end


See the tests included in the package for further examples.
Tests are located at 'test', 'swig/ruby/test', etc.


Known bugs
==============================

- The implementation of projectron is wrong.  It has to be fixed referring [#]_ and [#]_

.. [#] http://portal.acm.org/citation.cfm?id=1755875
.. [#] http://portal.acm.org/citation.cfm?id=1390247

.. Local variables:
.. mode: rst
.. End:

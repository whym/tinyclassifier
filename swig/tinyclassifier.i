%module tinyclassifier
%{
 #include "tinyclassifier.h"
 #include "perceptron.h"
%}

%include "stdint.i"
%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(BoolVector) vector<bool>;
  %template(IntVectorVector) vector<vector<int> >;
};

%include util.h
%template(power_int)   my_power<size_t>;
%template(power_float) my_power<double>;
%include tinyclassifier.h
%include perceptron.h


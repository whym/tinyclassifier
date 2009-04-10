%module TinyClassifier
%{
 #include "tinyclassifier.h"
 #include "perceptron.h"
%}

%include "stdint.i"
%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(FloatVector) vector<double>;
  %template(BoolVector) vector<bool>;
  %template(IntVectorVector) vector<vector<int> >;
  %template(FloatVectorVector) vector<vector<double> >;
};

%include util.h
%template(power_int)   my_power<size_t>;
%template(power_float) my_power<double>;
%include tinyclassifier.h
%include perceptron.h
%template(IntPerceptron) Perceptron<int>;
%template(FloatPerceptron) Perceptron<double>;
%template(IntPKPerceptron) PKPerceptron<int>;
%template(FloatPKPerceptron) PKPerceptron<double>;

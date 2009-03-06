%module tinyclassifier
%{
 #include "tinyclassifier.h"
 #include "perceptron.h"
%}

%include "stdint.i"
%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
};

%include tinyclassifier.h
%include perceptron.h

%inline %{
 inline void *memmove(void *dest, const char *src, size_t n) {
  return memmove(dest, reinterpret_cast<const void*>(src), n);
 }
%}

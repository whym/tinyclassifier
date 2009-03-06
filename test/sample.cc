#include <iostream>
#include "tinyclassifier.h"
#include "perceptron.h"
#include <vector>

int main() {
  Perceptron* x(new Perceptron(10));
  std::vector<feature_value_t> v;
  
  x->kernel(v,v);
  return 0;
}

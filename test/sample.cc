#include <iostream>
#include "tinyclassifier.h"
#include "perceptron.h"
#include <vector>

int main() {
  Perceptron x(Perceptron(3));
  std::vector<feature_value_t> v;
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);
  real_t result = x.kernel(v,v);
  std::cout << result << std::endl;
  if ( result == 10*10 + 20*20 + 30*30 ) {
    return 0;
  } else {
    return 1;
  }
}

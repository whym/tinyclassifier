#include <vector>
#include <iostream>

#include "perceptron.h"
#include "tinyclassifier.h"
#include "util.h"

Perceptron::Perceptron(size_t size_)
  : feature_vectors(), weights(), threshold(), size(size_) {
  std::cout << "Perceptron()" << std::endl;
}

Perceptron::~Perceptron() {
}

real_t Perceptron::kernel(std::vector<feature_value_t> v,
                          std::vector<feature_value_t> w) const {
  real_t ret = 0.0;
  VAR(it1, v.begin());
  VAR(it2, w.begin());
  for (size_t i = 0; i < size; ++i ) {
    if (it1 >= v.end() || it2 >= w.end()) {
      std::cerr << "vector index out of bounds: " << i << std::endl;
      return real_t(0);
    }
    ret += *it1 * *it2;
    ++it1;
    ++it2;
  }
  return ret;
}


void Perceptron::train(std::vector<std::vector<feature_value_t> > samples,
           std::vector<bool> sample_labels) {
}

bool Perceptron::predict(std::vector<feature_value_t> v) const {
  return true;
}

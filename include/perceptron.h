#ifndef _TINYCLASSIFIER_PERCEPTRON_H
#define _TINYCLASSIFIER_PERCEPTRON_H !NULL

#include <vector>
#include "tinyclassifier.h"

class Perceptron {
public:
  std::vector<std::vector<feature_value_t> > feature_vectors;
  std::vector<real_t> weights;
  real_t threshold;
  size_t iterations;
  size_t size;

  Perceptron(size_t);
  ~Perceptron();
  void train(std::vector<std::vector<feature_value_t> >, std::vector<bool>);
  real_t kernel(std::vector<feature_value_t>,
                std::vector<feature_value_t>) const;

  bool predict(std::vector<feature_value_t>) const;

private:
};

#endif // _TINYCLASSIFIER_PERCEPTRON_H

#ifndef _TINYCLASSIFIER_PERCEPTRON_H
#define _TINYCLASSIFIER_PERCEPTRON_H !NULL

#include <vector>
#include "tinyclassifier.h"

class Perceptron {
public:
  size_t feature_vector_size;
  size_t iterations;
  real_t delta;
  real_t threshold;
  size_t kernel_order;
  feature_value_t kernel_bias;
private:
  std::vector<real_t> weights;
  std::vector<real_t> weights_averaged;
  real_t bias;
  real_t bias_averaged;
  size_t averaging_count;

public:
  Perceptron(size_t size_, size_t iter=10, real_t d=1.0, size_t order=2, feature_value_t bias=1);
  ~Perceptron();

  void init();
  void train0(const std::vector<std::vector<feature_value_t> >&, const std::vector<bool>&);
  void train(const std::vector<std::vector<feature_value_t> >&, const std::vector<bool>&);
  feature_value_t kernel(const std::vector<feature_value_t>&,
                         const std::vector<feature_value_t>&) const;

  real_t predict0(const std::vector<feature_value_t>&) const;
  real_t predict(const std::vector<feature_value_t>&) const;

private:
  bool check_feature_vector(const std::vector<feature_value_t>&) const;
};

#endif // _TINYCLASSIFIER_PERCEPTRON_H

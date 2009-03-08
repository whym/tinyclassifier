#include <vector>
#include <iostream>
#include <numeric>

#include "perceptron.h"
#include "tinyclassifier.h"
#include "util.h"

Perceptron::Perceptron(size_t size_, size_t iter, real_t d, size_t order, feature_value_t bias)
  : feature_vector_size(size_), iterations(iter), delta(d), kernel_order(order), kernel_bias(bias) {
  init();
}
Perceptron::~Perceptron() {
}

void Perceptron::init() {
  this->weights_averaged.clear();
  this->weights.clear();

  for ( size_t i = 0; i < this->feature_vector_size; ++i ) {
    this->weights.push_back(0);
    this->weights_averaged.push_back(0);
  }
  this->averaging_count = 1;
  this->bias = 0.0;
  this->bias_averaged = 0.0;
}

feature_value_t Perceptron::kernel(const std::vector<feature_value_t>& v,
                                   const std::vector<feature_value_t>& w) const {
  if ( !this->check_feature_vector(v) ||
       !this->check_feature_vector(w) ) {
    return static_cast<feature_value_t>(0);
  }
  real_t ret = std::inner_product(v.begin(), v.begin()+feature_vector_size, w.begin(), this->kernel_bias);
  ret = my_power(ret, this->kernel_order);
  return ret;
}

void Perceptron::train0(const std::vector<std::vector<feature_value_t> >& samples,
                        const std::vector<bool>& sample_labels) {
  this->init();
  for ( size_t i = 0; i < this->iterations; ++i ) {
    bool no_change = true;
    for ( size_t j = 0; j < samples.size(); ++j ) {
      bool given_label = sample_labels[j];
      real_t prediction = this->predict0(samples[j]);
      if (!( (given_label && prediction > 0) || (!given_label && prediction < 0) )) {
        no_change = false;
        real_t given_polarity = given_label? +1.0: -1.0;
        for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
          this->weights[k] += given_polarity * this->delta * samples[j][k];
          this->weights_averaged[k] += given_polarity * this->delta * this->averaging_count * samples[j][k];
          this->bias += given_polarity;
          this->bias_averaged += given_polarity * this->averaging_count;
        }
      }
      ++this->averaging_count;
#ifdef DEBUG
      std::cout << this->averaging_count <<' '<< this->bias;
      print_range(std::cout, this->weights);
      std::cout << std::endl;
#endif
    }
    if ( no_change ) {          // TODO: ignore the updates within threshold
      break;
    }
  }

  for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
    this->weights[k] = this->weights[k] - static_cast<real_t>(this->weights_averaged[k]) / static_cast<real_t>(this->averaging_count);
  }
}

void Perceptron::train(const std::vector< std::vector< feature_value_t> >& samples,
                       const std::vector< bool >& sample_labels) {
}
 
real_t Perceptron::predict0(const std::vector<feature_value_t>& v) const {
  return std::inner_product(v.begin(), v.begin() + this->feature_vector_size, this->weights.begin(), this->bias);
}

real_t Perceptron::predict(const std::vector<feature_value_t>& v) const {
  real_t ret(0);
  return ret;
}

bool Perceptron::check_feature_vector(const std::vector<feature_value_t>& v) const {
  if ( v.size() == this->feature_vector_size ) {
    return true;
  } else {
    std::cerr << "vector index out of bounds: vector size is " << v.size() << std::endl;
    return false;
  }
}

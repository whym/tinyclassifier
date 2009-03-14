#ifndef _TINYCLASSIFIER_PERCEPTRON_H
#define _TINYCLASSIFIER_PERCEPTRON_H !NULL

#include <vector>
#include "tinyclassifier.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <set>

#include "perceptron.h"
#include "tinyclassifier.h"
#include "util.h"

template <typename feature_value_t, typename real_t=double, typename polarity_t=int> class Perceptron {
public:
  size_t feature_vector_size;
  size_t iterations;
  real_t delta;
  real_t threshold;
  size_t kernel_order;
  feature_value_t kernel_bias;
private:
  std::vector<real_t> weights;
  std::vector<real_t> weights_avg;
  real_t bias;
  real_t bias_avg;
  size_t averaging_count;
  std::set<std::vector<feature_value_t> > bases;
  std::vector<std::pair<real_t, typename std::set<std::vector<feature_value_t> >::const_iterator> > weighted_bases;
  std::vector<std::pair<real_t, typename std::set<std::vector<feature_value_t> >::const_iterator> > weighted_bases_avg;
 // TODO: 重複したサンプルの重みも共有すべき（現状は

public:

  Perceptron(size_t size_, size_t iter=10, real_t d=1.0, size_t order=2, feature_value_t bias=1)
    : feature_vector_size(size_), iterations(iter), delta(d), kernel_order(order), kernel_bias(bias) {
    init();
  }
  ~Perceptron() {
  }
  
  void init() {
    this->weights_avg.clear();
    this->weights.clear();
    this->weighted_bases.clear();
    this->weighted_bases_avg.clear();
    this->bases.clear();
    
    for ( size_t i = 0; i < this->feature_vector_size; ++i ) {
      this->weights.push_back(0);
      this->weights_avg.push_back(0);
    }
    this->averaging_count = 1;
    this->bias = 0.0;
    this->bias_avg = 0.0;
  }
  
  feature_value_t kernel(const std::vector<feature_value_t>& v,
                         const std::vector<feature_value_t>& w) const {
    if ( !this->check_feature_vector(v) ||
         !this->check_feature_vector(w) ) {
      return static_cast<feature_value_t>(0);
    }
    return my_power(std::inner_product(v.begin(), v.begin()+feature_vector_size, w.begin(), this->kernel_bias), this->kernel_order);
  }

  void train0(const std::vector<std::vector<feature_value_t> >& samples,
              const std::vector<polarity_t>& sample_labels) {
    this->init();
    for ( size_t i = 0; i < this->iterations; ++i ) {
#ifdef DEBUG
      std::cout << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg;
      std::cout << this->weights;
      std::cout << this->weights_avg;
      std::cout << std::endl;
#endif
      bool no_change = true;
      for ( size_t j = 0; j < samples.size(); ++j ) {
        real_t given_polarity = sample_labels[j];
        real_t prediction = this->predict0_avg(samples[j]);
        if (! (given_polarity * prediction > 0) ) { // TODO: ignore the difference within threshold
          no_change = false;
          for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
            this->weights[k]     += given_polarity * this->delta * samples[j][k];
            this->weights_avg[k] += given_polarity * this->delta * samples[j][k] * this->averaging_count;
          }
          this->bias     += given_polarity;
          this->bias_avg += given_polarity * this->averaging_count;
        }
        ++this->averaging_count;
      }
      if ( no_change ) {
        //break;
      }
    }
    
    for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
      this->weights[k] -= static_cast<real_t>(this->weights_avg[k]) / static_cast<real_t>(this->averaging_count);
    }
    this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);
  }
  
  real_t predict0_avg(const std::vector<feature_value_t>& v) const {
    // note that this is not normalized by averaging_count
    return std::inner_product(v.begin(), v.begin() + this->feature_vector_size,
                              this->weights.begin(),
                              this->bias) * this->averaging_count -
      std::inner_product(v.begin(), v.begin() + this->feature_vector_size,
                         this->weights_avg.begin(),
                         this->bias_avg);
  }

  real_t predict0(const std::vector<feature_value_t>& v) const {
    return std::inner_product(v.begin(), v.begin() + this->feature_vector_size,
                              this->weights.begin(),
                              this->bias);
  }
  
  void train(const std::vector< std::vector< feature_value_t> >& samples,
             const std::vector< polarity_t >& sample_labels) {
    this->init();
    FOREACH(it, samples) {
      VAR(p, this->bases.insert(*it));
      this->weighted_bases.push_back(std::make_pair(static_cast<real_t>(0), p.first));
      this->weighted_bases_avg.push_back(std::make_pair(static_cast<real_t>(0), p.first));
    }
    for ( size_t i = 0; i < this->iterations; ++i ) {
#ifdef DEBUG
      std::cout <<"train: " << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg;
      std::cout << this->weighted_bases;
      std::cout << std::endl;
#endif
      bool no_change = true;
      for ( size_t j = 0; j < samples.size(); ++j ) {
        VAR(given_polarity, sample_labels[j]);
        VAR(prediction, this->predict(samples[j]));
        if (! (given_polarity * prediction > 0) ) { // TODO: ignore the difference within threshold
          no_change = false;
          this->weighted_bases[j].first += static_cast<real_t>(this->delta) * given_polarity;
          this->weighted_bases_avg[j].first += static_cast<real_t>(this->delta) * given_polarity * this->averaging_count;
          this->bias     += static_cast<real_t>(given_polarity);
          this->bias_avg += static_cast<real_t>(given_polarity) * this->averaging_count;
        }
        ++this->averaging_count;
      }
      if ( no_change ) {
        //break;
      }
    }
    
    for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
      this->weighted_bases[k].first -= static_cast<real_t>(this->weighted_bases_avg[k].first) / static_cast<real_t>(this->averaging_count);
    }
    this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);
  }
  
  real_t predict(const std::vector<feature_value_t>& v) const {
    real_t ret = static_cast<real_t>(this->bias);
    FOREACH(it, this->weighted_bases){
      PAIRREF(w, bvecp, *it);
      ret += w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret *= this->averaging_count;
    FOREACH(it, this->weighted_bases_avg){
      PAIRREF(w, bvecp, *it);
      ret += w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret -= this->bias_avg;
    return ret;
  }
  
  bool check_feature_vector(const std::vector<feature_value_t>& v) const {
    if ( v.size() == this->feature_vector_size ) {
      return true;
    } else {
      std::cerr << "vector index out of bounds: vector size is " << v.size() << std::endl;
      return false;
    }
  }
};

#endif // _TINYCLASSIFIER_PERCEPTRON_H

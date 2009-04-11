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
#include "lru_cache.h"

template <typename feature_value_t, typename real_t=double, typename polarity_t=int, typename delta_t=int>
class Perceptron {
public:
  size_t feature_vector_size;
  size_t iterations;
  delta_t delta;
  bool check_convergence;
protected:
  std::vector<real_t> weights;
  std::vector<real_t> weights_avg;
  real_t bias;
  real_t bias_avg;
  size_t averaging_count;

public:

  Perceptron(size_t size_, size_t iter=40)
    : feature_vector_size(size_), iterations(iter),
      delta(1), check_convergence(true) {
    init();
  }
  ~Perceptron() {
  }
  
  void init() {
    this->weights_avg.clear();
    this->weights.clear();
    
    for ( size_t i = 0; i < this->feature_vector_size; ++i ) {
      this->weights.push_back(0);
      this->weights_avg.push_back(0);
    }
    this->averaging_count = 1;
    this->bias = 0.0;
    this->bias_avg = 0.0;
  }
  
  size_t train(const std::vector<std::vector<feature_value_t> >& samples,
                const std::vector<polarity_t>& sample_labels) {
    this->init();
    FOREACH(it, samples) check_feature_vector(*it);

    size_t i;
    for ( i = 0; i < this->iterations; ++i ) {
#ifdef DEBUG
      std::cout << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg;
      std::cout << this->weights;
      std::cout << this->weights_avg;
      std::cout << std::endl;
#endif
      bool no_change = true;
      for ( size_t j = 0; j < samples.size(); ++j ) {
        real_t given_polarity = sample_labels[j];
        real_t prediction = this->predict(samples[j]);
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
      if ( no_change && this->check_convergence ) {
        break;
      }
    }

    // TODO: これやったほうが公開用predictを2倍のはやさにできる
    // if do below, predict() must not do the same
//     for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
//       this->weights[k] -= static_cast<real_t>(this->weights_avg[k]) / static_cast<real_t>(this->averaging_count);
//     }
//     this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);
    return i;
  }
  
  real_t predict(const std::vector<feature_value_t>& v) const {
    check_feature_vector(v);    // TODO: faster if no check, for internal
    // NOTE: note that this value is not normalized by averaging_count
    return std::inner_product(v.begin(), v.begin() + this->feature_vector_size,
                              this->weights.begin(),
                              this->bias) * this->averaging_count -
      std::inner_product(v.begin(), v.begin() + this->feature_vector_size,
                         this->weights_avg.begin(),
                         this->bias_avg);
  }

protected:
  bool check_feature_vector(const std::vector<feature_value_t>& v) const {
    VAR(d, v.size() - this->feature_vector_size);
    if ( d == 0 ) {
      return true;
    } else if ( d > 0 ) {
#ifndef DEBUG
      return true;
#endif
    }
    std::cerr << "check_feature_vector(): expected " << this->feature_vector_size << ", but " << v.size() << std::endl;
    return false;
  }
};

// for base pointer
template<typename T>
std::ostream& operator<<(std::ostream& s, const typename std::set<std::vector<T> >::const_pointer& x) {
  return s << x << "=" << *x;
}

template <typename feature_value_t, typename real_t=double, typename polarity_t=int, typename delta_t=int> class PKPerceptron :public Perceptron<feature_value_t, real_t, polarity_t, delta_t> {
public:
  size_t kernel_order;
  feature_value_t kernel_bias;
private:
  typedef typename std::set<std::vector<feature_value_t> >::const_pointer base_pointer_type;
  std::set<std::vector<feature_value_t> > bases;
  std::vector<std::pair<delta_t, base_pointer_type> > weighted_bases;
  std::vector<std::pair<delta_t, base_pointer_type> > weighted_bases_avg;
  mutable      LRUCache<std::pair<base_pointer_type,base_pointer_type>, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t> > cache;
  friend class LRUCache<std::pair<base_pointer_type,base_pointer_type>, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t> >;
 // TODO: 重複したサンプルの重みも共有すべき？（現状は別々に扱う）

public:

  PKPerceptron(size_t size_, size_t iter=40, size_t order=2, feature_value_t bias=1)
    : Perceptron<feature_value_t, real_t, polarity_t>(size_, iter), kernel_order(order),
      kernel_bias(bias), cache(*this, 1024) {
    init();
  }
  ~PKPerceptron() {
  }
  
  void init() {
    this->weighted_bases.clear();
    this->weighted_bases_avg.clear();
    this->bases.clear();
    this->averaging_count = 1;
    this->bias = 0.0;
    this->bias_avg = 0.0;
    this->cache.init();
  }

  void set_cach_size(size_t i) {
    this->cache.set_size(i);
  }

private:
  feature_value_t produce(const std::pair<base_pointer_type,base_pointer_type>& p) const {
    return kernel(*(p.first), *(p.second));
  }
  
public:
  feature_value_t kernel(const std::vector<feature_value_t>& v,
                         const std::vector<feature_value_t>& w) const {
    return my_power(std::inner_product(v.begin(), v.begin() + this->feature_vector_size, w.begin(), this->kernel_bias), this->kernel_order);
  }

  size_t train(const std::vector< std::vector< feature_value_t> >& samples,
               const std::vector< polarity_t >& sample_labels) {
    this->init();
    FOREACH(it, samples) check_feature_vector(*it);

    FOREACH(it, samples) {
      VAR(p, this->bases.insert(*it));
      VAR(basep, &(*p.first));
      this->weighted_bases.push_back(std::make_pair(static_cast<real_t>(0), basep));
      this->weighted_bases_avg.push_back(std::make_pair(static_cast<real_t>(0), basep));
    }
    size_t i;
    for ( i = 0; i < this->iterations; ++i ) {
#ifdef DEBUG
      std::cout <<"train: " << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg << this->weighted_bases << std::endl;
#endif
      bool no_change = true;
      for ( size_t j = 0; j < samples.size(); ++j ) {
        VAR(given_polarity, sample_labels[j]);
        VAR(prediction, this->predict_of_base((this->weighted_bases[j].second)));
#ifdef DEBUG
        if ( this->predict(*(this->weighted_bases[j].second)) != prediction ) {
          std::cout << "predict and predict_of_base dont't match" << std::endl;
        }
#endif
        if (! (given_polarity * prediction > 0) ) { // TODO: ignore the difference within threshold
          no_change = false;
          this->weighted_bases[j].first += static_cast<delta_t>(this->delta) * given_polarity;
          this->weighted_bases_avg[j].first += static_cast<delta_t>(this->delta) * given_polarity * this->averaging_count;
          this->bias     += static_cast<real_t>(given_polarity);
          this->bias_avg += static_cast<real_t>(given_polarity) * this->averaging_count;
        }
        ++this->averaging_count;
      }
      if ( no_change && this->check_convergence ) {
#ifdef DEBUG
        std::cout << "terminate" << std::endl;
#endif
        break;
      }
    }
    
//     for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
//       this->weighted_bases[k].first -= static_cast<real_t>(this->weighted_bases_avg[k].first) / static_cast<real_t>(this->averaging_count);
//     }
//     this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);

    FOREACH(it, this->weighted_bases){
      VAR(w, it->first);
      if ( w == 0 ) {
        this->weighted_bases.erase(it);
      }
    }
    FOREACH(it, this->weighted_bases_avg){
      VAR(w, it->first);
      if ( w == 0 ) {
        this->weighted_bases_avg.erase(it);
      }
    }
    return i;
  }
  
  real_t predict(const std::vector<feature_value_t>& v) const {
    check_feature_vector(v);    // TODO: faster if no check, for internal
    // TODO: cache kernel values
    // NOTE: note that this value is not normalized by averaging_count
    real_t ret = static_cast<real_t>(this->bias);
    FOREACH(it, this->weighted_bases){
      PAIRREF(w, bvecp, *it);
      if ( w != 0 )             // TODO: not type safe
        ret += w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret *= this->averaging_count;
    FOREACH(it, this->weighted_bases_avg){
      PAIRREF(w, bvecp, *it);
      if ( w != 0 )
        ret -= w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret -= this->bias_avg;
    return ret;
  }

private:
  real_t predict_of_base(base_pointer_type basep) const {
    real_t ret = static_cast<real_t>(this->bias);
    FOREACH(it, this->weighted_bases){
      PAIRREF(w, bvecp, *it);
      if ( w != 0 ) {
        ret += w * static_cast<real_t>(this->cache.get(make_pair(bvecp, basep)));
      }
    }
    ret *= this->averaging_count;
    FOREACH(it, this->weighted_bases_avg){
      PAIRREF(w, bvecp, *it);
      if ( w != 0 ) {
        ret -= w * static_cast<real_t>(this->cache.get(make_pair(bvecp, basep)));
      }
    }
    ret -= this->bias_avg;
    return ret;
  }

};

#endif // _TINYCLASSIFIER_PERCEPTRON_H

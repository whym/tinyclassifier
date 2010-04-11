#ifndef _TINYCLASSIFIER_PERCEPTRON_H
#define _TINYCLASSIFIER_PERCEPTRON_H !NULL

#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <set>
#include <algorithm>
#include "util.h"
#include "lru_cache.h"

#define EPSILON 1.0E-4
#include <math.h>
template<typename value_t>
inline bool is_zero(value_t x, value_t eps=EPSILON) {
  return fabs(x) < eps;
}

template<>
inline bool is_zero(int x, int eps) {
  return x == 0;
}

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
  virtual ~Perceptron() {
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

  const char* store(const char* filename) {
    std::ofstream out(filename);
    out << this->bias << '\t' << this->bias_avg << std::endl;
    out << this->averaging_count << std::endl;
    for ( size_t i = 0; i < this->feature_vector_size; ++i ) {
      out << this->weights[i] << '\t' << this->weights_avg[i] << std::endl;
    }
    return filename;
  }
  
  void load(const char* filename) {
    this->init();
    std::ifstream in(filename);
    in >> this->bias;
    in >> this->bias_avg;
    in >> this->averaging_count;
    for ( size_t i = 0; i < this->averaging_count - 1; ++i ) {
      in >> this->weights[i];
      in >> this->weights_avg[i];
    }
  }
  
  size_t train(const std::vector<std::vector<feature_value_t> >& samples,
                const std::vector<polarity_t>& sample_labels) {
    this->init();
    FOREACH(it, samples) check_feature_vector(*it);

    size_t i;
    for ( i = 0; i < this->iterations; ++i ) {
#ifdef DEBUG
      std::cerr << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg;
      std::cerr << this->weights;
      std::cerr << this->weights_avg;
      std::cerr << std::endl;
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
  typedef size_t cache_size_t;
  real_t projection_threshold;
private:
  typedef typename std::set<std::vector<feature_value_t> >::const_pointer base_pointer_type;
  std::set<std::vector<feature_value_t> > bases;
  std::vector<base_pointer_type> base_pointers;
  std::vector<std::pair<delta_t, size_t> > weighted_bases;
  std::vector<std::pair<delta_t, size_t> > weighted_bases_avg;
  std::vector<real_t> norms;
  std::vector<feature_value_t> normal;
  typedef unsigned long cache_key_t;
  mutable      LRUCache<cache_key_t, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t> > cache;
  friend class LRUCache<cache_key_t, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t> >;
 // TODO: 重複したサンプルの重みも共有すべき？（現状は別々に扱う）

public:

  PKPerceptron(size_t size_, size_t iter=40, size_t order=2, feature_value_t bias=1, cache_size_t cache_size=0, real_t pth=0)
    : Perceptron<feature_value_t, real_t, polarity_t, delta_t>(size_, iter), kernel_order(order),
      kernel_bias(bias), projection_threshold(pth), cache(*this, cache_size) {
    init();
  }
  virtual ~PKPerceptron() {
  }
  
  void init() {
    this->weighted_bases.clear();
    this->weighted_bases_avg.clear();
    this->bases.clear();
    this->averaging_count = 1;
    this->bias = 0.0;
    this->bias_avg = 0.0;
    this->cache.init();
    this->base_pointers.clear();
    this->norms.clear();
  }

  void set_cache_size(cache_size_t i) const {
    this->cache.set_size(i);
  }

  int get_cache_size() const {
    return this->cache.get_size();
  }

private:
  inline size_t decode_x(cache_key_t p) const {
    return p / this->base_pointers.size();
    //return p.first;
  }
  inline size_t decode_y(cache_key_t p) const {
    return p % this->base_pointers.size();
    //return p.second;
  }
  inline cache_key_t encode_xy(size_t x, size_t y) const {
    if ( x > y ) {
      size_t t = x;
      x = y;
      y = t;
    }
    return static_cast<cache_key_t>(x) * this->base_pointers.size() + y;
  }

  feature_value_t produce(const cache_key_t& p) const {
//     std::cerr << "produce(): 1, "  << this->base_pointers.size() << ": "<< p <<", "<<decode_x(p) << ", " << decode_y(p) << std::endl; //!
//     std::cerr << (this->base_pointers[decode_x(p)]) << std::endl;//!
//     std::cerr << (this->base_pointers[decode_y(p)]) << std::endl;//!
//     std::cerr << *(this->base_pointers[decode_x(p)]) << std::endl;//!
//     std::cerr << *(this->base_pointers[decode_y(p)]) << std::endl;//!

    feature_value_t v = kernel(*(this->base_pointers[decode_x(p)]),
                               *(this->base_pointers[decode_y(p)]));
    return v;
  }

  std::vector<real_t> projection(const std::vector<feature_value_t>& x) const {
    std::vector<real_t> w(this->base_pointers.size(), 0);
    for ( size_t i = 0; i < w.size(); ++i ) {
      w[i] = std::inner_product(x.begin(), x.end(), this->base_pointers[i]->begin(), 0) / this->norms[i];
    }
    return w;
  }
  
public:
  feature_value_t kernel(const std::vector<feature_value_t>& v,
                         const std::vector<feature_value_t>& w) const {
    return my_power(std::inner_product(v.begin(), v.begin() + this->feature_vector_size, w.begin(), this->kernel_bias), this->kernel_order);
  }

  size_t train(const std::vector< std::vector< feature_value_t> >& samples,
               const std::vector< polarity_t >& sample_labels) {
    this->init();
    FOREACH(it, samples) {
      check_feature_vector(*it);

      VAR(p, this->bases.insert(*it));
      VAR(basep, &(*(p.first)));
      this->weighted_bases.push_back(std::make_pair(static_cast<real_t>(0), this->base_pointers.size()));
      this->weighted_bases_avg.push_back(std::make_pair(static_cast<real_t>(0), this->base_pointers.size()));
      this->base_pointers.push_back(basep);
      this->norms.push_back(std::inner_product(it->begin(), it->end(), it->begin(), 0));
    }
    size_t i;
    for ( i = 0; i < this->iterations; ++i ) {
      bool no_change = true;
      for ( size_t j = 0; j < samples.size(); ++j ) {
        VAR(given_polarity, sample_labels[j]);
        VAR(base_index, this->weighted_bases[j].second);
        VAR(prediction, this->cache.get_size() == 0 ?
            this->predict(*(this->base_pointers[base_index])):
            this->predict_of_base(base_index));
#ifdef DEBUG
        {
          real_t b = this->predict(*(this->base_pointers[base_index]));
          if ( !is_zero(b - prediction) ) {
            std::cerr << "predict and predict_of_base don't match: " << b << ", " << prediction <<  std::endl;
          }
        }
#endif
        if (! (given_polarity * prediction > 0) ) { // TODO: ignore the difference within threshold
          no_change = false;

          // obtain the projection of the jth vector onto the current plane
          VAR(projw, this->projection(*(this->base_pointers[base_index])));
          // compute the difference between the projection and the jth
          projw[j] -= static_cast<delta_t>(this->delta) * given_polarity;
          VAR(wnorm, std::inner_product(projw.begin(), projw.end(), projw.begin(), 0));
          if ( wnorm < this->projection_threshold ) {
#ifdef DEBUG
            //std::cerr << "projected? " << wnorm << "<" << this->projection_threshold << " : " << projw << std::endl;
#endif
            projw[j] += static_cast<delta_t>(this->delta) * given_polarity;
            for ( size_t k = 0; k < this->weighted_bases.size(); ++k ) {
              if ( !is_zero(projw[k]) ) {
                this->weighted_bases[k].first     += projw[k] * static_cast<delta_t>(this->delta) * given_polarity;
                this->weighted_bases_avg[k].first += projw[k] * static_cast<delta_t>(this->delta) * given_polarity * this->averaging_count;
              }
            }
          } else {
            this->weighted_bases[j].first     += static_cast<delta_t>(this->delta) * given_polarity;
            this->weighted_bases_avg[j].first += static_cast<delta_t>(this->delta) * given_polarity * this->averaging_count;
          }
          this->bias     += static_cast<real_t>(given_polarity);
          this->bias_avg += static_cast<real_t>(given_polarity) * this->averaging_count;
        }
        ++this->averaging_count;
      }
      if ( no_change && this->check_convergence ) {
#ifdef DEBUG
        std::cerr << "terminate" << std::endl;
#endif
        break;
      }
    }
#ifdef DEBUG
    std::cerr <<"train: " << i <<' '<< this->averaging_count <<' '<< this->bias <<' '<< this->bias_avg << this->weighted_bases << std::endl;
#endif

#ifdef DEBUG
    {
      int i = 0;
      FOREACH(it, this->weighted_bases){
        if ( !is_zero(it->first) ) ++i;
      }
      std::cerr << "#nonzero = " << i << std::endl;
    }
#endif
    
//     for ( size_t k = 0; k < this->feature_vector_size; ++k ) {
//       this->weighted_bases[k].first -= static_cast<real_t>(this->weighted_bases_avg[k].first) / static_cast<real_t>(this->averaging_count);
//     }
//     this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);

    return i;
  }
  
  real_t predict(const std::vector<feature_value_t>& v) const {
    check_feature_vector(v);
    // NOTE: note that this value is not normalized by averaging_count
    real_t ret = static_cast<real_t>(this->bias);
    FOREACH(it, this->weighted_bases){
      PAIRREF(w, bveci, *it);
      VAR(bvecp, this->base_pointers[bveci]);
      if ( !is_zero(w) )
        ret += w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret *= this->averaging_count;
    FOREACH(it, this->weighted_bases_avg){
      PAIRREF(w, bveci, *it);
      VAR(bvecp, this->base_pointers[bveci]);
      if ( !is_zero(w) )
        ret -= w * static_cast<real_t>(kernel(*bvecp, v));
    }
    ret -= this->bias_avg;
    return ret;
  }

  const char* store(const char* filename) {
    std::ofstream out(filename);
    out << this->bias << '\t' << this->bias_avg << std::endl;
    out << this->base_pointers.size() << std::endl;
    out << this->weighted_bases.size() << std::endl;
    out << this->averaging_count << std::endl;
    FOREACH(it, this->base_pointers) {
      //! bases でまとめられている同じベクトルを複数回書き出すのが無駄
      print_range(out, **it, "\t", "", "\n");
    }
    for ( size_t i = 0; i < this->weighted_bases.size(); ++i ) {
      out << this->weighted_bases[i].first << '\t'
          << this->weighted_bases[i].second << '\t'
          << this->weighted_bases_avg[i].first << '\t'
          << this->weighted_bases_avg[i].second << std::endl;
    }
    return filename;
  }
  
  void load(const char* filename) {
    this->init();
    std::ifstream in(filename);
    in >> this->bias;
    in >> this->bias_avg;
    size_t basenum, weightnum;
    in >> basenum;
    in >> weightnum;;
    in >> this->averaging_count;
    for ( size_t i = 0; i < basenum; ++i ) {
      std::vector<feature_value_t> vec;
      for ( size_t j = 0; j < this->feature_vector_size; ++j ) {
        feature_value_t x;
        in >> x;
        vec.push_back(x);
      }
      VAR(p, this->bases.insert(vec));
      VAR(basep, &(*(p.first)));
      this->base_pointers.push_back(basep);
      this->norms.push_back(std::inner_product(basep->begin(), basep->end(), basep->begin(), 0));
    }
    for ( size_t i = 0; i < weightnum; ++i ) {
      delta_t x1, x2;
      size_t y1,y2;
      in >> x1;
      in >> y1;
      in >> x2;
      in >> y2;
      this->weighted_bases.push_back(std::make_pair(x1, y1));
      this->weighted_bases_avg.push_back(std::make_pair(x2, y2));
    }
  }

private:
  real_t predict_of_base(size_t basei) const {
    real_t ret = static_cast<real_t>(this->bias);
    FOREACH(it, this->weighted_bases){
      PAIRREF(w, bveci, *it);
      if ( !is_zero(w) ) {
        ret += w * static_cast<real_t>(this->cache.get(encode_xy(bveci, basei)));
      }
    }
    ret *= this->averaging_count;
    FOREACH(it, this->weighted_bases_avg){
      PAIRREF(w, bveci, *it);
      if ( !is_zero(w) ) {
        ret -= w * static_cast<real_t>(this->cache.get(encode_xy(bveci, basei)));
      }
    }
    ret -= this->bias_avg;
    return ret;
  }

};

#endif // _TINYCLASSIFIER_PERCEPTRON_H

#ifdef TEST_TINYCLASSIFIER_PERCEPTRON_H
#undef TEST_TINYCLASSIFIER_PERCEPTRON_H

#include "util.h"
#include <vector>
#include <functional>
#include <cstdio>
#include <iostream>

using namespace std;

bool succeed = true;

static void ok(bool b, const char* name = "") {
  static int n = 1;
  printf("%s %d - %s\n", b ? "ok" : "ng", n++, name);
  succeed = succeed && b;
}
template <typename T> void is(const T& x, const T& y, const char* name = "") {
  if (x == y) {
    ok(true, name);
  } else {
    ok(false, name);
    cout << " expected: " << x << ", given: " << y << endl;
  }
}
template <typename T, typename S> void same_sign(const T& x, const S& y, const char* name = "") {
  if (x * y > 0) {
    ok(true, name);
  } else {
    ok(false, name);
    cout << " expected: " << x << ", given: " << y << endl;
  }
}
template <typename T> void equal(T f1, T l1, T f2, T l2, const char* name = "") {
  if (equal(f1, l1, f2)) {
    ok(true, name);
  } else {
    ok(false, name);
    cout << " expected: ";
    print_range(cout, f1, l1);
    cout <<", given: ";
    print_range(cout, f2, l2);
    cout << "}" << endl;
  }
}

#include <sstream>
#include <ctime>
#include <cstdio>

const char* temp_path() {
  std::stringstream s;
  s << "f57a" << time(NULL) << "__";
  return s.str().c_str();
}

int main() {
  typedef int feature_value_t;
  typedef double real_t;

  {
    PKPerceptron<feature_value_t, real_t, int, real_t> kperc(3);
    kperc.iterations = 20;
    vector<feature_value_t> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    VAR(result, kperc.kernel(v,v));
    
    is(my_power(10*10 + 20*20 + 30*30 + 1, 2), result, "perceptron#polynomial_kernel");
  }

  {
    PKPerceptron<feature_value_t, real_t, int, real_t> kperc(3);
    Perceptron<feature_value_t, real_t> perc(3);
    perc.iterations = 20;
    vector<vector<feature_value_t> > samples;
    vector<feature_value_t> v;
    v.clear(); v.push_back(+1); v.push_back(+1); v.push_back(-1);
    samples.push_back(v);
    
    v.clear(); v.push_back(-2); v.push_back(+2); v.push_back(+1);
    samples.push_back(v);
    
    v.clear(); v.push_back(-1); v.push_back(+3); v.push_back(+4);
    samples.push_back(v);
    
    v.clear(); v.push_back(-2); v.push_back(+5); v.push_back(+6);
    samples.push_back(v);
    
    v.clear(); v.push_back(-1); v.push_back(+1); v.push_back(+1);
    samples.push_back(v);
    
    v.clear(); v.push_back(-1); v.push_back(+1); v.push_back(+1);
    samples.push_back(v);
    
    vector<int> b;
    b.push_back(-1);
    b.push_back(-1);
    b.push_back(+1);
    b.push_back(+1);
    b.push_back(-1);
    b.push_back(-1);
    
    perc.train(samples, b);
    Perceptron<feature_value_t, real_t> perc2(3);
    const char* temp = temp_path();
    perc.store(temp);
    perc2.load(temp);
    remove(temp);
    for ( size_t i = 0; i < samples.size(); ++i ) {
      real_t res = perc.predict(samples[i]);
      same_sign(b[i], res,
                (string("perceptron_perceptron#") + STR(i)).c_str());
      same_sign(res, perc2.predict(samples[i]),
                (string("perceptron_load_store#") + STR(i)).c_str());
    }

    kperc.train(samples, b);
    PKPerceptron<feature_value_t, real_t, int, real_t> kperc2(3);
    temp = temp_path();
    kperc.store(temp);
    kperc2.load(temp);
    remove(temp);
    for ( size_t i = 0; i < samples.size(); ++i ) {
      real_t res = kperc.predict(samples[i]);
      same_sign(b[i], res,
                (string("perceptron_with_kernel#") + STR(i)).c_str());
      same_sign(res, kperc2.predict(samples[i]),
                (string("perceptron_with_kernel_load_store#") + STR(i)).c_str());
    }
    kperc.kernel_order = 3;
    kperc.kernel_bias = 1;
    kperc.projection_threshold = 1;
    kperc.iterations = 20000;
    kperc.check_convergence = false;
    kperc.set_cache_size(1000);
    int its = kperc.train(samples, b);
    cout << "iterations = " << its << endl;
    for ( size_t i = 0; i < samples.size(); ++i ) {
      real_t res = kperc.predict(samples[i]);
      same_sign(b[i], res, (string("perceptron_with_cache_projection#") + STR(i)).c_str());
    }
  }
  return !succeed;
}

#endif

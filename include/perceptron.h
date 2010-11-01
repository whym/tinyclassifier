#ifndef _TINYCLASSIFIER_PERCEPTRON_H
#define _TINYCLASSIFIER_PERCEPTRON_H !NULL

#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cassert>
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

template<typename value_t>
inline bool is_nan(value_t x) {
  return x != x;
}
template<>
inline bool is_nan(int x) {
  return false;
}

template<typename value_t>
bool read_with_default(std::istream& i, value_t& x, const value_t& dvalue=0) {
  i >> x;
  if ( i.fail() ) {
    x = dvalue;
    i.clear();
    i.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return false;
  } else {
    return true;
  }
}

template <typename feature_value_t, typename real_t=double, typename polarity_t=int, typename delta_t=feature_value_t, typename predict_t=feature_value_t>
class Perceptron {
protected:
  size_t dimensions;
public:
  size_t iterations;
  delta_t delta;
  bool check_convergence;
protected:
  std::vector<feature_value_t> weights;
  std::vector<feature_value_t> weights_avg;
  polarity_t bias;
  polarity_t bias_avg;
  int averaging_count;

public:

  Perceptron(size_t dim, size_t iter=40)
    : dimensions(dim), iterations(iter),
      delta(1), check_convergence(true),
      weights(dim, 0), weights_avg(dim, 0),
      bias(0), bias_avg(0),
      averaging_count(1) {
    init();
  }
  virtual ~Perceptron() {
  }

  size_t get_dimensions() { return this->dimensions; }
  void   set_dimensions(size_t n) { this->dimensions = n; }

  void init() {
    init(this->dimensions, this->iterations);
  }
  void init(size_t dim) {
    init(dim, this->iterations);
  }
  void init(size_t dim, size_t iter) {
    this->dimensions = dim;
    this->iterations = iter;
    this->weights_avg.clear();
    this->weights.clear();
    this->weights_avg.resize(this->dimensions, 0);
    this->weights.resize(this->dimensions, 0);
    this->bias = 0.0;
    this->bias_avg = 0.0;
    this->averaging_count = 1;
  }

  const char* store(const char* filename) {
    std::ofstream out(filename);
    out << this->bias << '\t' << this->bias_avg << std::endl;
    out << this->averaging_count << std::endl;
    for ( size_t i = 0; i < this->dimensions; ++i ) {
      out << this->weights[i] << '\t' << this->weights_avg[i] << std::endl;
    }
    return filename;
  }
  
  bool load(const char* filename) {
    this->init();
    std::ifstream in(filename);

    if ( !read_with_default(in, this->bias) ||
         !read_with_default(in, this->bias_avg) ||
         !read_with_default(in, this->averaging_count, 1) ) {
      return false;
    }
    for ( size_t i = 0; i < this->dimensions; ++i ) {
      if ( in.eof() ) {
        return false;
      }
      if ( !read_with_default(in, this->weights[i])
           || !read_with_default(in, this->weights_avg[i]) ) {
        return false;
      }
    }
    return true;
  }
  
  size_t train(const std::vector<std::vector<feature_value_t> >& samples,
               const std::vector<polarity_t>& sample_labels) {
    FOREACH(it, samples) {
       if ( it->size() < this->dimensions ) {
         this->dimensions = it->size();
       }
      check_feature_vector(*it);
    }
    this->init();

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
        polarity_t given_polarity = sample_labels[j];
        feature_value_t prediction = this->predict0(samples[j]);
        assert(!is_nan(prediction));
        assert(!is_nan(given_polarity));
        if (! (prediction * given_polarity > 0) ) { // TODO: ignore the difference within threshold
          no_change = false;
          for ( size_t k = 0; k < this->dimensions; ++k ) {
            this->weights[k]     += (given_polarity * this->delta) * samples[j][k];
            this->weights_avg[k] += (given_polarity * this->delta * this->averaging_count) * samples[j][k];
          }
          this->bias     += given_polarity;
          this->bias_avg += given_polarity * this->averaging_count;
        }
        ++this->averaging_count;
      }
      if ( no_change && this->check_convergence ) {
        break;
      }
      for ( size_t k = 0; k < this->dimensions; ++k ) {
        if ( is_nan(this->weights[k]) ) {
          std::cerr << "nan at [" << k << "]" << std::endl;
          this->weights[k] = 0;
        }
        if ( is_nan(this->weights_avg[k]) ) {
          std::cerr << "nan at [" << k << "] avg" << std::endl;
          this->weights_avg[k] = 0;
        }
      }
    }
    return i;
  }

  predict_t predict(const std::vector<feature_value_t>& v) const {
    check_feature_vector(v);
    VAR(ret, this->predict0(v));
    return ret / predict_t(this->averaging_count);
  }

protected:
  feature_value_t predict0(const std::vector<feature_value_t>& v) const {
    assert(v.size() >= this->weights.size());
    assert(v.size() >= this->weights_avg.size());
    assert(!is_nan(this->bias));
    assert(!is_nan(this->bias_avg));
    
    return feature_value_t(this->averaging_count)
      * std::inner_product(v.begin(), v.begin() + this->dimensions,
                           this->weights.begin(),
                           feature_value_t(this->bias))
      - std::inner_product(v.begin(), v.begin() + this->dimensions,
                           this->weights_avg.begin(),
                           feature_value_t(this->bias_avg));
    // TODO:
    // public version of predict() doesn't have to use two weights; it
    // canbe merged to half the computation.
  }

protected:
  bool check_feature_vector(const std::vector<feature_value_t>& v) const {
    VAR(d, v.size() - this->dimensions);
    if ( d == 0 ) {
      return true;
    } else if ( d > 0 ) {
#ifndef DEBUG
      return true;
#endif
    }
    std::cerr << "check_feature_vector(): expected " << this->dimensions << ", but " << v.size() << std::endl;
    return false;
  }
};

// for base pointer
template<typename T>
std::ostream& operator<<(std::ostream& s, const typename std::set<std::vector<T> >::const_pointer& x) {
  return s << x << "=" << *x;
}

template <typename feature_value_t, typename real_t=double, typename polarity_t=int, typename delta_t=feature_value_t, typename predict_t=real_t> class PKPerceptron :public Perceptron<feature_value_t, real_t, polarity_t, delta_t, predict_t> {
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
  std::vector<std::vector<real_t> > orthonormals;
  typedef unsigned long cache_key_t;
  mutable      LRUCache<cache_key_t, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t, predict_t> > cache;
  friend class LRUCache<cache_key_t, feature_value_t, PKPerceptron<feature_value_t, real_t, polarity_t, delta_t, predict_t> >;
 // TODO: 重複したサンプルの重みも共有すべき？（現状は別々に扱う）
  typedef Perceptron<feature_value_t, real_t, polarity_t, delta_t, predict_t> super_type_t;

public:

  PKPerceptron(size_t dim, size_t iter=40, size_t order=2, feature_value_t bias=1, cache_size_t cache_size=0, real_t pth=0)
    : Perceptron<feature_value_t, real_t, polarity_t, delta_t, predict_t>(dim, iter), kernel_order(order),
      kernel_bias(bias), projection_threshold(pth), cache(*this, cache_size) {
    init();
  }
  virtual ~PKPerceptron() {
  }

  size_t get_dimensions() { return super_type_t::get_dimensions(); }
  void   set_dimensions(size_t n) { super_type_t::set_dimensions(n); }

  void init() {
    super_type_t::init(this->dimensions);
    this->weighted_bases.clear();
    this->weighted_bases_avg.clear();
    this->bases.clear();
    this->averaging_count = 1;
    this->bias = 0.0;
    this->bias_avg = 0.0;
    this->cache.init();
    this->base_pointers.clear();
    this->norms.clear();
    this->orthonormals.clear();
  }

  void set_cache_size(cache_size_t i) const {
    this->cache.set_size(i);
  }

  int get_cache_size() const {
    return this->cache.get_size();
  }

private:
  template<typename Val> inline Val norm(const std::vector<Val>& x, Val v = 0) {
    return std::inner_product(x.begin(), x.end(), x.begin(), v);
  }
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

  std::vector<real_t> projection_residue(const std::vector<feature_value_t>& x) const {
    std::vector<real_t> w(x.size());
    std::copy(x.begin(), x.end(), w.begin());
    for ( size_t i = 1; i < this->orthonormals.size(); ++i ) {
      VAR(len, std::inner_product(w.begin(), w.end(), this->orthonormals[i].begin(), 0));
      for ( size_t j = 0; j < w.size(); ++j ) {
        w[j] -= len * this->orthonormals[i][j];
      }
    }
    //std::cerr << x << ", " << w << std::endl;//!
    return w;
  }
  
public:
  feature_value_t kernel(const std::vector<feature_value_t>& v,
                         const std::vector<feature_value_t>& w) const {
    return my_power(std::inner_product(v.begin(), v.begin() + this->dimensions, w.begin(), this->kernel_bias), this->kernel_order);
  }

  size_t train(const std::vector< std::vector< feature_value_t> >& samples,
               const std::vector< polarity_t >& sample_labels) {
    FOREACH(it, samples) {
       if ( it->size() < this->dimensions ) {
         this->dimensions = it->size();
       }
      check_feature_vector(*it);
    }
    this->init();
    FOREACH(it, samples) {
      check_feature_vector(*it);

      VAR(p, this->bases.insert(*it));
      VAR(basep, &(*(p.first)));
      this->weighted_bases.push_back(std::make_pair(static_cast<real_t>(0), this->base_pointers.size()));
      this->weighted_bases_avg.push_back(std::make_pair(static_cast<real_t>(0), this->base_pointers.size()));
      this->base_pointers.push_back(basep);
      this->norms.push_back(norm(*it));
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
          //std::cerr << "projw: " <<  projw << std::endl;//!
          VAR(residue, projection_residue(*(this->base_pointers[base_index])));
          //std::cerr << "resid: " <<  residue << std::endl;//!
          VAR(rnorm, norm(residue));
          if ( rnorm < this->projection_threshold ) {
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
            // add a new orthonormal
            {
              FOREACH(it, residue) {
                *it = *it / rnorm;
              }
              this->orthonormals.push_back(residue);
            }            
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
    
//     for ( size_t k = 0; k < this->dimensions; ++k ) {
//       this->weighted_bases[k].first -= static_cast<real_t>(this->weighted_bases_avg[k].first) / static_cast<real_t>(this->averaging_count);
//     }
//     this->bias -= static_cast<real_t>(this->bias_avg) / static_cast<real_t>(this->averaging_count);

    return i;
  }
  
  predict_t predict(const std::vector<feature_value_t>& v) const {
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
  
  bool load(const char* filename) {
    this->init();
    std::ifstream in(filename);
    if ( !read_with_default(in, this->bias) ||
         !read_with_default(in, this->bias_avg) ) {
      return false;
    }
    size_t basenum, weightnum;
    if ( !read_with_default<size_t>(in, basenum, 1) ||
         !read_with_default<size_t>(in, weightnum, 1) ||
         !read_with_default(in, this->averaging_count, 1) ) {
      return false;
    }
    for ( size_t i = 0; i < basenum; ++i ) {
      std::vector<feature_value_t> vec;
      for ( size_t j = 0; j < this->dimensions; ++j ) {
        feature_value_t x;
        if ( !read_with_default(in, x) ) {
          return false;
        }
        vec.push_back(x);
      }
      VAR(p, this->bases.insert(vec));
      VAR(basep, &(*(p.first)));
      this->base_pointers.push_back(basep);
      this->norms.push_back(norm(*basep));
    }
    for ( size_t i = 0; i < weightnum; ++i ) {
      delta_t x1, x2;
      size_t y1,y2;
      if ( !read_with_default(in, x1) ||
           !read_with_default(in, y1) ||
           !read_with_default(in, x2) ||
           !read_with_default(in, y2 ) ) {
        return false;
      }
      this->weighted_bases.push_back(std::make_pair(x1, y1));
      this->weighted_bases_avg.push_back(std::make_pair(x2, y2));
    }
    return true;
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

std::string temp_path() {
  std::stringstream s;
  s << "f57a" << time(NULL) << "__";
  return s.str();
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
    Perceptron<feature_value_t, real_t, int, feature_value_t, real_t> perc(3);
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
    
    ok(perc.train(samples, b) == 1, "percetpron_perceptron#train");
    Perceptron<feature_value_t, real_t, int, feature_value_t, real_t> perc2(3);
    string temp = temp_path();
    ok(perc.store(temp.c_str()), string("percetpron_perceptron#store " + temp).c_str());
    bool ok_load = perc2.load(temp.c_str());
    ok(ok_load, "percetpron_perceptron#load");
    if ( ok_load ) {
      remove(temp.c_str());
    } else {
      return 1;
    }
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
    kperc.store(temp.c_str());
    kperc2.load(temp.c_str());
    remove(temp.c_str());
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

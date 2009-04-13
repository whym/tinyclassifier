#include <iostream>
#include "tinyclassifier.h"
#include "perceptron.h"
#include "util.h"
#include "lru_cache.h"
#include <vector>
#include <functional>

struct Func {
  int produce(const std::pair<int,int>& p) const { return p.first * p.first + p.second; }
};

namespace std {
 namespace tr1 {
   template<>
   struct hash<std::pair<int, int> > {
     size_t operator()(const std::pair<int,int>& p) const {
       return p.first * 10001 + p.second;
     }
   };
 }
}

int main() {
  using namespace std;
  LRUCache<pair<int,int>, int, Func> cache(Func(), 3);
  bool hit;
  cache.get(make_pair(100,100), hit);  cout << hit << endl;
  cache.get(make_pair(10,10), hit);  cout << hit << endl;
  cache.get(make_pair(100,100), hit);  cout << hit << endl;
  cache.get(make_pair(2,2), hit);  cout << hit << endl;
  cache.get(make_pair(1,1), hit);  cout << hit << endl;

  if ( 10*10*10*10*10 != my_power(10,5) ) {
    cout << my_power(10,5) << endl;
    return 1;
  }
  if ( 10*10*10 != my_power(10,3) ) {
    cout << my_power(10,3) << endl;
    return 1;
  }
  typedef int feature_value_t;
  typedef double real_t;

  Perceptron<feature_value_t, real_t> perc(3);
  PKPerceptron<feature_value_t, real_t> kperc(3);
  perc.iterations = 20;
  kperc.iterations = 20;
  vector<feature_value_t> v;
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);
  VAR(result, kperc.kernel(v,v));
  std::cout << result << std::endl;
  if ( result != my_power(10*10 + 20*20 + 30*30 + 1, 2) ) {
    return 1;
  }
  vector<vector<feature_value_t> > samples;

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

  vector<int> b;
  b.push_back(-1);
  b.push_back(-1);
  b.push_back(+1);
  b.push_back(+1);
  b.push_back(-1);

  perc.train(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = perc.predict(samples[i]);
    cout << b[i] << "? " <<  res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  kperc.train(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = kperc.predict(samples[i]);
    cout << b[i] << "? " << res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  kperc.kernel_order = 3;
  kperc.kernel_bias = 1;
  kperc.iterations = 2000000;
  kperc.check_convergence = false;
  kperc.set_cache_size(1024*1024*100);
  kperc.set_cache_size(0);
  int its = kperc.train(samples, b);
  cout << "iterations = " << its << endl;
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = kperc.predict(samples[i]);
    cout << b[i] << "? " << res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  return 0;
}

#include <iostream>
#include "tinyclassifier.h"
#include "perceptron.h"
#include "util.h"
#include <vector>

int main() {
  using namespace std;

  if ( 10*10*10*10*10 != my_power(10,5) ) {
    cout << my_power(10,5) << endl;
    return 1;
  }
  if ( 10*10*10 != my_power(10,3) ) {
    cout << my_power(10,3) << endl;
    return 1;
  }
  typedef int feature_value_t;
  Perceptron<feature_value_t> perc(3);
  perc.iterations = 20;
  vector<feature_value_t> v;
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);
  VAR(result, perc.kernel(v,v));
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

  perc.train0(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = perc.predict0(samples[i]);
    cout << res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  perc.train(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = perc.predict(samples[i]);
    cout << res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  perc.kernel_order = 3;
  perc.kernel_bias = 1;
  perc.train(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = perc.predict(samples[i]);
    cout << res << endl;
    if ( b[i] * res < 0 ) {
      return 1;
    }
  }
  return 0;
}

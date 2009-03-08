#include <iostream>
#include "tinyclassifier.h"
#include "perceptron.h"
#include "util.h"
#include <vector>

int main() {
  using namespace std;
  Perceptron perc(3);
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

  v.clear();
  v.push_back(+1); v.push_back(+1); v.push_back(-1);
  samples.push_back(v);

  v.clear();
  v.push_back(-2); v.push_back(+2); v.push_back(+1);
  samples.push_back(v);

  v.clear();
  v.push_back(-1); v.push_back(+3); v.push_back(+4);
  samples.push_back(v);

  v.clear();
  v.push_back(-1); v.push_back(+5); v.push_back(+6);
  samples.push_back(v);

  v.clear();
  v.push_back(-1); v.push_back(+1); v.push_back(+2);
  samples.push_back(v);

  vector<bool> b;
  b.push_back(false);
  b.push_back(false);
  b.push_back(true);
  b.push_back(true);
  b.push_back(false);

  perc.train0(samples, b);
  for ( size_t i = 0; i < samples.size(); ++i ) {
    real_t res = perc.predict0(samples[i]);
    cout << res << endl;
    if ( (b[i]?+1:-1) * res < 0 ) {
      return 1;
    }
  }
  return 0;
}

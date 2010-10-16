#include <iostream>
#include <vector>
#include <boost/numeric/interval.hpp>
#include "perceptron.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, boost::numeric::interval<T> x) {
  return os << "(" << x.lower() << ", " << x.upper() << ")";
}

template<typename T>
std::vector<T> make_vec(const T& x, const T& y, const T& z) {
  std::vector<T> v;
  v.clear(); v.push_back(x); v.push_back(y); v.push_back(z);
  return v;
}

int main(){
  using namespace std;
  typedef boost::numeric::interval<double> feature_value_t;
  typedef double real_t;
  typedef double polarity_t;

  Perceptron<feature_value_t, real_t, polarity_t, real_t> perc(3);
  perc.iterations = 20;

  vector<vector<feature_value_t> > samples;

  samples.push_back(make_vec(feature_value_t(-2), feature_value_t(+2), feature_value_t(+1)));
  samples.push_back(make_vec(feature_value_t(+1), feature_value_t(+1), feature_value_t(-1)));
  samples.push_back(make_vec(feature_value_t(-1), feature_value_t(+3), feature_value_t(+4)));
  samples.push_back(make_vec(feature_value_t(-2), feature_value_t(+5), feature_value_t(+6)));
  samples.push_back(make_vec(feature_value_t(-1), feature_value_t(+1), feature_value_t(+1)));
  samples.push_back(make_vec(feature_value_t(-2), feature_value_t(+1), feature_value_t(+1)));
    
  vector<polarity_t> b;
  b.push_back(-1);
  b.push_back(-1);
  b.push_back(+1);
  b.push_back(+1);
  b.push_back(-1);
  b.push_back(-1);

  perc.train(samples, b);

  for ( size_t i = 0; i < b.size(); ++i ) {
    cout << b[i] << ":" << perc.predict(samples[i]) << endl;
    if ( perc.predict(samples[i]) * b[i] < 0 )
      return 1;
  }
  return 0;
}

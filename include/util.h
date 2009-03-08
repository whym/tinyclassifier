#ifndef _TINYCLASSIFIER_UTIL_H
#define _TINYCLASSIFIER_UTIL_H !NULL

#include <algorithm>
#include <iterator>
#include <iostream>

#define VAR(var, val)  typeof(val) var(val)
#define REF(var, val)  typeof(val)& var(val)
#define CREF(var, val) const typeof(val)& var(val)
#define FOREACH(iter, cntr) for(typeof((cntr).begin()) iter = (cntr).begin(); iter != (cntr).end(); ++iter)
// FIXME: rewrite with BOOST_FOREACH
#define PAIRREF(var1, var2, pair)\
  VAR(__tmp_pair, (pair));                                              \
  REF(var1, __tmp_pair.first);                                          \
  REF(var2, __tmp_pair.second);

template<typename T>
T my_power(T base, size_t degree) {
  T ret = 1;
  while (degree >= 2) {
    if (degree %2 == 1) {
      ret *= base;
    }
    base *= base; // base << 1; // for integer
    degree /= 2;
  }
  return base;
}

template<typename Range>
std::ostream& print_range(std::ostream& s,
                          const Range& x,
                          const char* delim  = ", ",
                          const char* lbrace = "{",
                          const char* rbrace = "}") {
  VAR(i, x.begin());
  VAR(last, x.end());
  if ( i == last ) {
    return s << lbrace << rbrace;
  } else {
    --last;
    s << lbrace;
    std::copy(i, last, std::ostream_iterator<typename Range::value_type>(s, delim));
    return s << *last << rbrace;
  }
}


#endif // _TINYCLASSIFIER_UTIL_H

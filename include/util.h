#ifndef _TINYCLASSIFIER_UTIL_H
#define _TINYCLASSIFIER_UTIL_H !NULL

#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <list>
#include <set>

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
  while (degree >= 1) {
    if ((degree & 1) == 1) {
      ret *= base;
    }
    base *= base;
    degree >>= 1;
 }
  return ret;
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
    for ( ;i != last; ++i ) {
      s << *i << delim;
    }
    return s << *last << rbrace;
  }
}

template<typename U, typename V>
std::ostream& operator<<(std::ostream& s, const std::pair<U,V>& p) {
  return s << "(" << p.first << "," << p.second << ")";
}
template<typename V>
std::ostream& operator<<(std::ostream& s, const std::vector<V>& x) {
  return print_range(s, x);
}

template<typename V>
std::ostream& operator<<(std::ostream& s, const std::list<V>& x) {
  return print_range(s, x);
}
template<typename V>
std::ostream& operator<<(std::ostream& s, const std::set<V>& x) {
  return print_range(s, x);
}

// iterator
template<typename I>
std::ostream& print_iterator_value(std::ostream& s, const I& it) {
  return s << "&(" << *it << ")";
}

#ifdef DEBUG
template<typename V>
std::ostream& operator<<(std::ostream& s, const typename std::vector<V>::const_iterator& x) {
  return print_iterator_value(s, x);
}
template<typename V>
std::ostream& operator<<(std::ostream& s, const typename std::list<V>::const_iterator& x) {
  return print_iterator_value(s, x);
}
template<typename V>
std::ostream& operator<<(std::ostream& s, const typename std::set<V>::const_iterator& x) {
  return print_iterator_value(s, x);
}
template<typename V>
std::ostream& operator<<(std::ostream& s, const std::_Rb_tree_const_iterator<V>& x) {
  return print_iterator_value(s, x);
}
#endif

#endif // _TINYCLASSIFIER_UTIL_H

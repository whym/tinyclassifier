#ifndef _TINYCLASSIFIER_UTIL_H
#define _TINYCLASSIFIER_UTIL_H !NULL

#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>
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
#define STR(s)                                                          \
  (dynamic_cast<std::ostringstream&>(std::ostringstream() << s)).str()


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

#ifdef TEST_TINYCLASSIFIER_UTIL_H
#undef TEST_TINYCLASSIFIER_UTIL_H
#include <iostream>
#include <cstdio>

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

int main() {
  {
    is(10*10*10*10*10, my_power(10,5), "my_power#1");
    is(10*10*10, my_power(10,3), "my_power#2");
  }

  {
    is(string("10"), STR(10), "STR#1");
    is(string("-10.2"), STR(-10.2), "STR#2");
    is(string("2 times"), STR(2 << " times"), "STR#3");
    is(string("at 2:30"), string("at ") + STR(2 << ":" << 30), "STR#4");
  }

  return !succeed;
}
#endif

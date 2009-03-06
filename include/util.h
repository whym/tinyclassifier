#ifndef _TINYCLASSIFIER_UTIL_H
#define _TINYCLASSIFIER_UTIL_H !NULL

#define VAR(var, val)  typeof(val) var(val)
#define REF(var, val)  typeof(val)& var(val)
#define CREF(var, val) const typeof(val)& var(val)
#define FOREACH(iter, cntr) for(typeof((cntr).begin()) iter = (cntr).begin(); iter != (cntr).end(); ++iter)
// FIXME: rewrite with BOOST_FOREACH
#define PAIRREF(var1, var2, pair)\
  VAR(__tmp_pair, (pair));                                              \
  REF(var1, __tmp_pair.first);                                          \
  REF(var2, __tmp_pair.second);

#endif // _TINYCLASSIFIER_UTIL_H

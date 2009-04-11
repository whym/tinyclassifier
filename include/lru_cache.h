#ifndef _TINYCLASSIFIER_LRUCACHE_H
#define _TINYCLASSIFIER_LRUCACHE_H !NULL

#include <map>
#include <list>
#include <functional>
#include <tr1/unordered_map>

// based on http://www.koders.com/cpp/fid4C1FB5C18106581BF78EB2720ED01C2EFC1DD340.aspx?s=cdef%3atree+mdef%3ainsert

// The requirement to user is:
// - factory_t should implement 'produce' function, which convert key_t
//   to value_t
// - key_t shuold be hashable/comparable, because we use it as map's key

template <class key_t, typename value_t, typename factory_t, typename cache_t=std::map<key_t,value_t> >  
class LRUCache {
private:
  size_t max;
  cache_t cache;
  std::list<key_t> queue;
  const factory_t& factory;

public:
  LRUCache(const factory_t& fact, size_t i = 1024) : max(i), factory(fact) {}

  void init() {
    queue.clear();
    cache.clear();
  }

  void set_size(size_t i) {
    max = i;
  }

  value_t get(const key_t& k) {
    bool tmp;
    return get(k, tmp);
  }
  value_t get(const key_t &k, bool& hit) {
    typename cache_t::iterator iter = cache.find(k);
    hit = (iter != cache.end());
    value_t v;
    if (hit) {
      v = iter->second;
    } else {
      v = factory.produce(k);
      cache.insert(std::make_pair(k,v));
      queue.push_front(k);
      if (queue.size() > max) {
        key_t victim = queue.back();
        cache.erase(victim);
        queue.pop_back();
      }
    }
    //std::cerr << "LRUCache: get(" << k << "), " << hit <<", value=" << v<< std::endl;
    return v;
  }
};

#endif // _TINYCLASSIFIER_LRUCACHE_H

#ifndef _TINYCLASSIFIER_LRUCACHE_H
#define _TINYCLASSIFIER_LRUCACHE_H !NULL

#include <map>
#include <list>
#include <functional>
#include <tr1/unordered_map>

// based on http://www.koders.com/cpp/fid4C1FB5C18106581BF78EB2720ED01C2EFC1DD340.aspx?s=cdef%3atree+mdef%3ainsert

// The requirements to user are:
// - factory_t should implement 'produce' function, which converts key_t
//   to value_t
// - key_t shuold be hashable/comparable, because we use it as map's key

template <class key_t, typename value_t, typename factory_t, typename cache_t=std::tr1::unordered_map<key_t,value_t> >
class LRUCache {
private:
  typedef unsigned int cache_size_t;
  cache_size_t max;
  cache_t cache;
  std::list<key_t> queue;
  const factory_t& factory;
  int hits;
  int misses;

public:
  LRUCache(const factory_t& fact, cache_size_t i = 1024) :
    max(i), factory(fact), hits(0), misses(0) {}

  ~LRUCache(){
#ifdef DEBUG
    std::cerr << "lru_cache: hit=" << hits << ", miss="<<misses<<std::endl;
#endif
  }

  void init() {
    queue.clear();
    cache.clear();
  }

  void set_size(cache_size_t i) {
    max = i;
  }

  cache_size_t get_size() const {
    return max;
  }

  inline value_t get(const key_t& k) {
    bool tmp;
    return get(k, tmp);
  }
  inline value_t get(const key_t &k, bool& hit) {
#ifdef DEBUG
    //    std::cerr << "LRUCache: get(" << k << "), " << (cache.find(k) != cache.end()?"hit":"miss") << std::endl;
#endif
    typename cache_t::iterator iter = cache.find(k);
    if (iter != cache.end()) {
      ++hits;
      return iter->second;
    } else {
      value_t v = factory.produce(k);
      cache.insert(std::make_pair(k,v));
      queue.push_front(k);
      if (queue.size() > max) {
        key_t victim = queue.back();
        cache.erase(victim);
        queue.pop_back();
      }
      ++misses;
      return v;
    }
  }
};

#endif // _TINYCLASSIFIER_LRUCACHE_H

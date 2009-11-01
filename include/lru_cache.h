#ifndef _TINYCLASSIFIER_LRUCACHE_H
#define _TINYCLASSIFIER_LRUCACHE_H !NULL

#include <map>
#include <list>
#include <functional>
#include <iostream>
#include <tr1/unordered_map>

// The requirements to user are:
// - factory_t should implement 'produce' function, which converts key_t
//   to value_t
// - key_t shuold be hashable/comparable, because it is used as map's key


template <class key_t, typename value_t, typename factory_t>
class LRUCache {
protected:
  typedef unsigned int cache_size_t;
  typedef typename std::list<std::pair<key_t, value_t> > queue_t;
  cache_size_t max;
  queue_t queue;
  const factory_t& factory;
  int hits;
  int misses;
private:
  typedef typename std::tr1::unordered_map<key_t, typename queue_t::iterator> map_t;
  map_t map;

public:
  LRUCache(const factory_t& fact, cache_size_t i = 1024) :
    max(i), factory(fact), hits(0), misses(0) {}

  virtual ~LRUCache(){
    queue.clear();
    map.clear();
#ifdef DEBUG
    std::cerr << "lru_cache: size=" << queue.size() << ", hit=" << hits << ", miss="<<misses<<std::endl;
#endif
  }

  virtual void init() {
    queue.clear();
    map.clear();
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
  inline value_t get(const key_t& k, bool& hit) {
#ifdef DEBUG
    //std::cerr << "LRUCache::get(" << k << "), " << (map.find(k) != map.end()?"hit":"miss") << std::endl;
#endif
    typename queue_t::iterator iter = find(k);
    if (iter != queue.end()) {
      ++hits;
      set_as_recently_used(iter);
      hit = true;
      return queue.begin()->second;
    } else {
      value_t v = factory.produce(k);
      set_as_recently_used(k, v);
      ++misses;
      hit = false;
      return v;
    }
  }
protected:
  inline virtual typename queue_t::iterator find(const key_t& k) {
    typename map_t::const_iterator i = map.find(k);
    if ( i == map.end() )
      return queue.end();
    else
      return i->second;
  }
  inline virtual void set_as_recently_used(const key_t& k, const value_t& v) {
    if ( max <= 0 ) {
      return;
    }
    typename queue_t::iterator tmp = queue.begin();
    queue.push_front(std::make_pair(k,v));
    map.insert(std::make_pair(k, queue.begin()));
    if ( queue.size() > max ) {
      key_t deleted_key = queue.back().first;
#ifdef DEBUG
      //std::cerr << "LRUCache::purge " << deleted_key << std::endl;
#endif
      queue.pop_back();
      map.erase(deleted_key);
    }
  }
  inline virtual void set_as_recently_used(const typename queue_t::iterator& it) {
    if ( it != queue.begin() ) {
      typename queue_t::value_type tmp = *(it);
      queue.erase(it);
      queue.push_front(tmp);
      map[tmp.first] = this->queue.begin();
    }
  }
};

template <typename key_t, typename value_t, typename factory_t>
class IntLRUCache : public LRUCache<key_t, value_t, factory_t> {
  // FIXME: don't pass test
private:
  typedef typename LRUCache<key_t, value_t, factory_t>::cache_size_t cache_size_t;
  typedef typename LRUCache<key_t, value_t, factory_t>::queue_t queue_t;
  typedef typename std::vector<typename queue_t::iterator> map_t;
  map_t map;

public:
  IntLRUCache(const factory_t& fact, cache_size_t i = 1024) : LRUCache<key_t, value_t, factory_t>(fact, i) {
  }

  virtual void init() {
    LRUCache<key_t, value_t, factory_t>::init();
    map.clear();
  }

protected:
  inline virtual typename queue_t::iterator find(const key_t& k) {
    if ( map.size() <= k )
      return this->queue.end();

    return map[k];
  }
  inline virtual void set_as_recently_used(const key_t&k, const value_t& v) {
    if ( this->max <= 0 ) {
      return;
    }
    typename queue_t::iterator tmp = this->queue.begin();
    this->queue.push_front(std::make_pair(k,v));
    map.reserve(k);
    map[k] = this->queue.begin();
    if ( this->queue.size() > this->max ) {
      key_t deleted_key = this->queue.back().first;
#ifdef DEBUG
      //std::cerr << "LRUCache::purge " << deleted_key << std::endl;
#endif
      this->queue.pop_back();
      map[deleted_key] = this->queue.end();
    }
  }
  inline virtual void set_as_recently_used(const typename queue_t::iterator& it) {
    if ( it != this->queue.begin() ) {
      typename queue_t::value_type tmp = *(it);
      this->queue.erase(it);
      this->queue.push_front(tmp);
      map[tmp.first] = this->queue.begin();
    }
  }
};

#endif // _TINYCLASSIFIER_LRUCACHE_H

#ifdef TEST_TINYCLASSIFIER_LRUCACHE_H
#undef TEST_TINYCLASSIFIER_LRUCACHE_H
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


struct Func {
  int produce(const std::pair<int,int>& p) const {
    return p.first * p.first + p.second;
  }
};

struct Square {
  int produce(int x) const {
    return x*x;
  }
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

  {
    list<int> ls;
    ls.push_front(1);
  }
  {
    Square t;
    LRUCache<int, int, Square> cache(t, 10);
    //IntLRUCache<int, int, Square> cache(t, 10);
    is(t.produce(10), cache.get(10), "lru_cache#1");
    is(t.produce(11), cache.get(11), "lru_cache#2");
    bool hit;
    is(t.produce(10), cache.get(10, hit), "lru_cache#3a");
    ok(hit, "lru_cache#3");
    is(t.produce(9999), cache.get(9999, hit), "lru_cache#4a");
    ok(!hit, "lru_cache#4");
  }

  {
    Func t;
    LRUCache<pair<int,int>, int, Func> cache(t, 3);
    bool hit;
    is(t.produce(make_pair(100,100)), cache.get(make_pair(100,100), hit), "lru_cache2#begin");
    ok(!hit);
    is(t.produce(make_pair(10,10)),cache.get(make_pair(10,10), hit));
    ok(!hit);
    is(t.produce(make_pair(100,100)), cache.get(make_pair(100,100), hit));
    ok(hit);
    is(t.produce(make_pair(2,2)), cache.get(make_pair(2,2), hit));
    ok(!hit, "lru_cache2#");
    is(t.produce(make_pair(1,1)), cache.get(make_pair(1,1), hit));
    ok(!hit);
    is(t.produce(make_pair(100,100)),cache.get(make_pair(100,100), hit));
    ok(hit);
    is(t.produce(make_pair(10,10)),cache.get(make_pair(10,10), hit));
    ok(!hit);
    is(t.produce(make_pair(1,1)), cache.get(make_pair(1,1), hit));
    ok(hit);
    is(t.produce(make_pair(2,2)), cache.get(make_pair(2,2), hit));
    ok(!hit, "lru_cache2#end");
  }
  {
    Func t;
    LRUCache<pair<int,int>, int, Func> cache(t, 31);
    bool hit;
    cache.get(make_pair(-1,-1));
    for ( int i = 1; i <= 30; ++i ) {
      cache.get(make_pair(i,i));
    }
    for ( int i = 30; i >= 1; --i ) {
      cache.get(make_pair(i,i));
    }
    is(t.produce(make_pair(-1,-1)), cache.get(make_pair(-1,-1), hit), "lru_cache3#1");
    ok(hit, "lru_cache3#2");
  }

  return !succeed;
}
#endif

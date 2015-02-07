// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-inefficient-algorithm %t
// REQUIRES: shell

namespace std {
template <typename T> struct less {
  bool operator()(const T &lhs, const T &rhs) { return lhs < rhs; }
};

template <typename T> struct greater {
  bool operator()(const T &lhs, const T &rhs) { return lhs > rhs; }
};

struct iterator_type {};

template <typename K, typename Cmp = less<K>> struct set {
  typedef iterator_type iterator;
  iterator find(const K &k);
  unsigned count(const K &k);

  iterator begin();
  iterator end();
  iterator begin() const;
  iterator end() const;
};

template <typename K> struct unordered_set : set<K> {};

template <typename K, typename Cmp = less<K>> struct multiset : set<K, Cmp> {};

template <typename FwIt, typename K> FwIt find(FwIt, FwIt, const K &);

template <typename FwIt, typename K, typename Cmp>
FwIt find(FwIt, FwIt, const K &, Cmp);

template <typename FwIt, typename K> FwIt count(FwIt, FwIt, const K &);

template <typename FwIt, typename K> FwIt lower_bound(FwIt, FwIt, const K &);
}

#define FIND_IN_SET(x) find(x.begin(), x.end(), 10) 
// CHECK-FIXES: #define FIND_IN_SET(x) find(x.begin(), x.end(), 10)

template <typename T> void f(const T &t) {
  std::set<int> s;
  find(s.begin(), s.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}s.find(46);{{$}}

  find(t.begin(), t.end(), 46);
  // CHECK-FIXES: {{^  }}find(t.begin(), t.end(), 46);{{$}}
}

int main() {
  std::set<int> s;
  auto it = std::find(s.begin(), s.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: this STL algorithm call should be replaced with a container method [misc-inefficient-algorithm]
  // CHECK-FIXES: {{^  }}auto it = s.find(43);{{$}}
  auto c = count(s.begin(), s.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}auto c = s.count(43);{{$}}

  std::multiset<int> ms;
  find(ms.begin(), ms.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}ms.find(46);{{$}}

  const std::multiset<int> &msref = ms;
  find(msref.begin(), msref.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}msref.find(46);{{$}}

  std::multiset<int> *msptr = &ms;
  find(msptr->begin(), msptr->end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}msptr->find(46);{{$}}

  it = std::find(s.begin(), s.end(), 43, std::greater<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: different comparers used in the algorithm and the container [misc-inefficient-algorithm]

  FIND_IN_SET(s);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}FIND_IN_SET(s);{{$}}

  f(s);

  std::unordered_set<int> us;
  lower_bound(us.begin(), us.end(), 10);
  // CHECK-FIXES: {{^  }}lower_bound(us.begin(), us.end(), 10);{{$}}
  find(us.begin(), us.end(), 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: {{^  }}us.find(10);{{$}}
}

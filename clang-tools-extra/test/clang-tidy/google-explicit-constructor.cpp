// RUN: $(dirname %s)/check_clang_tidy.sh %s google-explicit-constructor %t
// REQUIRES: shell

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

struct A {
  A() {}
  A(int x, int y) {}
  A(std::initializer_list<int> list1) {}

  explicit A(void *x) {}
  explicit A(void *x, void *y) {}

  explicit A(const A& a) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}A(const A& a) {}

  explicit A(std::initializer_list<double> list2) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}A(std::initializer_list<double> list2) {}

  A(int x1) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors must be explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit A(int x1) {}

  A(double x2, double y = 3.14) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors must be explicit
  // CHECK-FIXES: {{^  }}explicit A(double x2, double y = 3.14) {}
};

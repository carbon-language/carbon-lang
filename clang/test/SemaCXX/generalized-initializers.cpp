// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s
// XFAIL: *

template <typename T, typename U>
struct same_type { static const bool value = false; };
template <typename T>
struct same_type<T, T> { static const bool value = true; };

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

namespace integral {

  void initialization() {
    { const int a{}; static_assert(a == 0, ""); }
    { const int a = {}; static_assert(a == 0, ""); }
    { const int a{1}; static_assert(a == 1, ""); }
    { const int a = {1}; static_assert(a == 1, ""); }
    { const int a{1, 2}; } // expected-error {{ too many initializers}}
    { const int a = {1, 2}; } // expected-error {{ too many initializers}}
    { const short a = {100000}; } // expected-error {{narrowing conversion}}
  }

  int function_call() {
    void takes_int(int);
    takes_int({1});

    int ar[10];
    (void) ar[{1}]; // expected-error {{initializer list is illegal with the built-in index operator}}

    return {1};
  }

  void inline_init() {
    (void) int{1};
    (void) new int{1};
  }

  void initializer_list() {
    auto l = {1, 2, 3, 4};
    static_assert(same_type<decltype(l), std::initializer_list<int>>::value, "");

    for (int i : {1, 2, 3, 4}) {}
  }

  struct A {
    int i;
    A() : i{1} {}
  };

}

namespace objects {

  struct A {
    A();
    A(int, double);
    A(int, int);
    A(std::initializer_list<int>);

    int operator[](A);
  };

  void initialization() {
    // FIXME: how to ensure correct overloads are called?
    { A a{}; }
    { A a = {}; }
    { A a{1, 1.0}; }
    { A a = {1, 1.0}; }
    { A a{1, 2, 3, 4, 5, 6, 7, 8}; }
    { A a = {1, 2, 3, 4, 5, 6, 7, 8}; }
    { A a{1, 2, 3, 4, 5, 6, 7, 8}; }
    { A a{1, 2}; }
  }

  A function_call() {
    void takes_A(A);
    takes_a({1, 1.0});

    A a;
    a[{1, 1.0}];

    return {1, 1.0};
  }

  void inline_init() {
    (void) A{1, 1.0};
    (void) new A{1, 1.0};
  }

  struct B {
    B(A, int, A);
  };

  void nested_init() {
    B b{{1, 1.0}, 2, {3, 4, 5, 6, 7}};
  }
}

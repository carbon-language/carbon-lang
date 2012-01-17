// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

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

struct one { char c[1]; };
struct two { char c[2]; };

struct A {
  int a, b;
};

struct B {
  B();
  B(int, int);
};

void simple_list() {
  std::initializer_list<int> il = { 1, 2, 3 };
  std::initializer_list<double> dl = { 1.0, 2.0, 3 };
  std::initializer_list<A> al = { {1, 2}, {2, 3}, {3, 4} };
  std::initializer_list<B> bl = { {1, 2}, {2, 3}, {} };
}

void function_call() {
  void f(std::initializer_list<int>);
  f({1, 2, 3});

  void g(std::initializer_list<B>);
  g({ {1, 2}, {2, 3}, {} });
}

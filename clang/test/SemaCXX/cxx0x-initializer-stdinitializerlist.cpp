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

template <typename T, typename U>
struct same_type { static const bool value = false; };
template <typename T>
struct same_type<T, T> { static const bool value = true; };

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

struct C {
  C(int);
};

struct D {
  D();
  operator int();
  operator C();
};

void overloaded_call() {
    one overloaded(std::initializer_list<int>);
    two overloaded(std::initializer_list<B>);

    static_assert(sizeof(overloaded({1, 2, 3})) == sizeof(one), "bad overload");
    static_assert(sizeof(overloaded({ {1, 2}, {2, 3}, {} })) == sizeof(two), "bad overload");

    void ambiguous(std::initializer_list<A>); // expected-note {{candidate}}
    void ambiguous(std::initializer_list<B>); // expected-note {{candidate}}
    ambiguous({ {1, 2}, {2, 3}, {3, 4} }); // expected-error {{ambiguous}}

    one ov2(std::initializer_list<int>); // expected-note {{candidate}}
    two ov2(std::initializer_list<C>); // expected-note {{candidate}}
    // Worst sequence to int is identity, whereas to C it's user-defined.
    static_assert(sizeof(ov2({1, 2, 3})) == sizeof(one), "bad overload");
    // But here, user-defined is worst in both cases.
    ov2({1, 2, D()}); // expected-error {{ambiguous}}
}

template <typename T>
T deduce(std::initializer_list<T>); // expected-note {{conflicting types for parameter 'T' ('int' vs. 'double')}}
template <typename T>
T deduce_ref(const std::initializer_list<T>&); // expected-note {{conflicting types for parameter 'T' ('int' vs. 'double')}}

void argument_deduction() {
  static_assert(same_type<decltype(deduce({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce({1, 2.0}); // expected-error {{no matching function}}

  static_assert(same_type<decltype(deduce_ref({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce_ref({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce_ref({1, 2.0}); // expected-error {{no matching function}}
}

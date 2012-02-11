// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct one { char c[1]; };
struct two { char c[2]; };

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

namespace objects {

  struct X1 { X1(int); };
  struct X2 { explicit X2(int); }; // expected-note 2 {{candidate constructor}}

  template <int N>
  struct A {
    A() { static_assert(N == 0, ""); }
    A(int, double) { static_assert(N == 1, ""); }
  };

  template <int N>
  struct F {
    F() { static_assert(N == 0, ""); }
    F(int, double) { static_assert(N == 1, ""); }
    F(std::initializer_list<int>) { static_assert(N == 3, ""); }
  };

  template <int N>
  struct D {
    D(std::initializer_list<int>) { static_assert(N == 0, ""); } // expected-note 1 {{candidate}}
    D(std::initializer_list<double>) { static_assert(N == 1, ""); } // expected-note 1 {{candidate}}
  };

  template <int N>
  struct E {
    E(int, int) { static_assert(N == 0, ""); }
    E(X1, int) { static_assert(N == 1, ""); }
  };

  void overload_resolution() {
    { A<0> a{}; }
    { A<0> a = {}; }
    { A<1> a{1, 1.0}; }
    { A<1> a = {1, 1.0}; }

    { F<0> f{}; }
    { F<0> f = {}; }
    // Narrowing conversions don't affect viability. The next two choose
    // the initializer_list constructor.
    // FIXME: Emit narrowing conversion errors.
    { F<3> f{1, 1.0}; } // xpected-error {{narrowing conversion}}
    { F<3> f = {1, 1.0}; } // xpected-error {{narrowing conversion}}
    { F<3> f{1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f = {1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f{1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f{1, 2}; }

    { D<0> d{1, 2, 3}; }
    { D<1> d{1.0, 2.0, 3.0}; }
    { D<-1> d{1, 2.0}; } // expected-error {{ambiguous}}

    { E<0> e{1, 2}; }
  }

  void explicit_implicit() {
    { X1 x{0}; }
    { X1 x = {0}; }
    { X2 x{0}; }
    { X2 x = {0}; } // expected-error {{no matching constructor}}
  }

  struct C {
    C();
    C(int, double);
    C(int, int);

    int operator[](C);
  };

  C function_call() {
    void takes_C(C);
    takes_C({1, 1.0});

    C c;
    c[{1, 1.0}];

    return {1, 1.0};
  }

  void inline_init() {
    (void) C{1, 1.0};
    (void) new C{1, 1.0};
    (void) A<1>{1, 1.0};
    (void) new A<1>{1, 1.0};
  }

  struct B { // expected-note 2 {{candidate constructor}}
    B(C, int, C); // expected-note {{candidate constructor not viable: cannot convert initializer list argument to 'objects::C'}}
  };

  void nested_init() {
    B b1{{1, 1.0}, 2, {3, 4}};
    B b2{{1, 1.0, 4}, 2, {3, 4}}; // expected-error {{no matching constructor for initialization of 'objects::B'}}
  }

  void overloaded_call() {
    one ov1(B); // expected-note {{not viable: cannot convert initializer list}}
    two ov1(C); // expected-note {{not viable: cannot convert initializer list}}

    static_assert(sizeof(ov1({})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({1, 2})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({{1, 1.0}, 2, {3, 4}})) == sizeof(one), "bad overload");

    ov1({1}); // expected-error {{no matching function}}

    one ov2(int);
    two ov2(F<3>);
    static_assert(sizeof(ov2({1})) == sizeof(one), "bad overload"); // list -> int ranks as identity
    static_assert(sizeof(ov2({1, 2, 3})) == sizeof(two), "bad overload"); // list -> F only viable
  }
}

// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

namespace dr1460 { // dr1460: 3.5
#if __cplusplus >= 201103L
  namespace DRExample {
    union A {
      union {};
      union {};
      constexpr A() {}
    };
    constexpr A a = A();

    union B {
      union {};
      union {};
      constexpr B() = default;
    };
    constexpr B b = B();

    union C {
      union {};
      union {};
    };
    constexpr C c = C();
#if __cplusplus > 201103L
    constexpr void f() { C c; }
    static_assert((f(), true), "");
#endif
  }

  union A {};
  union B { int n; }; // expected-note +{{here}}
  union C { int n = 0; };
  struct D { union {}; };
  struct E { union { int n; }; }; // expected-note +{{here}}
  struct F { union { int n = 0; }; };

  struct X {
    friend constexpr A::A() noexcept;
    friend constexpr B::B() noexcept; // expected-error {{follows non-constexpr declaration}}
    friend constexpr C::C() noexcept;
    friend constexpr D::D() noexcept;
    friend constexpr E::E() noexcept; // expected-error {{follows non-constexpr declaration}}
    friend constexpr F::F() noexcept;
  };

  // These are OK, because value-initialization doesn't actually invoke the
  // constructor.
  constexpr A a = A();
  constexpr B b = B();
  constexpr C c = C();
  constexpr D d = D();
  constexpr E e = E();
  constexpr F f = F();

  namespace Defaulted {
    union A { constexpr A() = default; };
    union B { int n; constexpr B() = default; }; // expected-error {{not constexpr}}
    union C { int n = 0; constexpr C() = default; };
    struct D { union {}; constexpr D() = default; };
    struct E { union { int n; }; constexpr E() = default; }; // expected-error {{not constexpr}}
    struct F { union { int n = 0; }; constexpr F() = default; };

    struct G { union { int n = 0; }; union { int m; }; constexpr G() = default; }; // expected-error {{not constexpr}}
    struct H {
      union {
        int n = 0;
      };
      union { // expected-note 2{{member not initialized}}
        int m;
      };
      constexpr H() {} // expected-error {{must initialize all members}}
      constexpr H(bool) : m(1) {}
      constexpr H(char) : n(1) {} // expected-error {{must initialize all members}}
      constexpr H(double) : m(1), n(1) {}
    };
  }

#if __cplusplus > 201103L
  template<typename T> constexpr bool check() {
    T t; // expected-note-re 2{{non-constexpr constructor '{{[BE]}}'}}
    return true;
  }
  static_assert(check<A>(), "");
  static_assert(check<B>(), ""); // expected-error {{constant}} expected-note {{in call}}
  static_assert(check<C>(), "");
  static_assert(check<D>(), "");
  static_assert(check<E>(), ""); // expected-error {{constant}} expected-note {{in call}}
  static_assert(check<F>(), "");
#endif

  union G {
    int a = 0; // expected-note {{previous initialization is here}}
    int b = 0; // expected-error {{initializing multiple members of union}}
  };
  union H {
    union {
      int a = 0; // expected-note {{previous initialization is here}}
    };
    union {
      int b = 0; // expected-error {{initializing multiple members of union}}
    };
  };
  struct I {
    union {
      int a = 0; // expected-note {{previous initialization is here}}
      int b = 0; // expected-error {{initializing multiple members of union}}
    };
  };
  struct J {
    union { int a = 0; };
    union { int b = 0; };
  };

  namespace Overriding {
    struct A {
      int a = 1, b, c = 3;
      constexpr A() : b(2) {}
    };
    static_assert(A().a == 1 && A().b == 2 && A().c == 3, "");

    union B {
      int a, b = 2, c;
      constexpr B() : a(1) {}
      constexpr B(char) : b(4) {}
      constexpr B(int) : c(3) {}
      constexpr B(const char*) {}
    };
    static_assert(B().a == 1, "");
    static_assert(B().b == 2, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(B('x').a == 0, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(B('x').b == 4, "");
    static_assert(B(123).b == 2, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(B(123).c == 3, "");
    static_assert(B("").a == 1, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(B("").b == 2, "");
    static_assert(B("").c == 3, ""); // expected-error {{constant}} expected-note {{read of}}

    struct C {
      union { int a, b = 2, c; };
      union { int d, e = 5, f; };
      constexpr C() : a(1) {}
      constexpr C(char) : c(3) {}
      constexpr C(int) : d(4) {}
      constexpr C(float) : f(6) {}
      constexpr C(const char*) {}
    };

    static_assert(C().a == 1, "");
    static_assert(C().b == 2, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C().d == 4, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C().e == 5, "");

    static_assert(C('x').b == 2, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C('x').c == 3, "");
    static_assert(C('x').d == 4, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C('x').e == 5, "");

    static_assert(C(1).b == 2, "");
    static_assert(C(1).c == 3, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C(1).d == 4, "");
    static_assert(C(1).e == 5, ""); // expected-error {{constant}} expected-note {{read of}}

    static_assert(C(1.f).b == 2, "");
    static_assert(C(1.f).c == 3, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C(1.f).e == 5, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C(1.f).f == 6, "");

    static_assert(C("").a == 1, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C("").b == 2, "");
    static_assert(C("").c == 3, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C("").d == 4, ""); // expected-error {{constant}} expected-note {{read of}}
    static_assert(C("").e == 5, "");
    static_assert(C("").f == 6, ""); // expected-error {{constant}} expected-note {{read of}}

    struct D;
    extern const D d;
    struct D {
      int a;
      union {
        int b = const_cast<D&>(d).a = 1; // not evaluated
        int c;
      };
      constexpr D() : a(0), c(0) {}
    };
    constexpr D d {};
    static_assert(d.a == 0, "");
  }
#endif
}

#if __cplusplus >= 201103L
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

  template < class _T1, class _T2 > struct pair { _T2 second; };

  template<typename T> struct basic_string {
    basic_string(const T* x) {}
    ~basic_string() {};
  };
  typedef basic_string<char> string;

} // std

namespace dr1467 {  // dr1467: yes c++11
  // List-initialization of aggregate from same-type object
  
  namespace basic0 {
    
    struct S {
      int i = 42;
    };
    
    S a;
    S b(a);
    S c{a};
    
    struct SS : public S { } x;
    S y(x);
    S z{x};
    
  } // basic0
  
  namespace basic1 {
    
    struct S {
      int i{42};
    };
    
    S a;
    S b(a);
    S c{a};
    
    struct SS : public S { } x;
    S y(x);
    S z{x};
    
  } // basic1
  
  namespace basic2 {
    
    struct S {
      int i = {42};
    };
    
    S a;
    S b(a);
    S c{a};
    
    struct SS : public S { } x;
    S y(x);
    S z{x};
    
  } // basic2
  
  namespace dr_example {
    struct OK {
      OK() = default;
      OK(const OK&) = default;
      OK(int) { }
    };
    
    OK ok;
    OK ok2{ok};
    
    
    struct X {
      X() = default;
      X(const X&) = default;
    };
    
    X x;
    X x2{x};
  } // dr_example
  
  namespace nonaggregate {
    
    struct NonAggregate {
      NonAggregate() {}
    };
    
    struct WantsIt {
      WantsIt(NonAggregate);
    };
    
    void f(NonAggregate);
    void f(WantsIt);
    
    void test1() {
      NonAggregate n;
      f({n});
    }

    void test2() {
      NonAggregate x;
      NonAggregate y{x};
      NonAggregate z{{x}};
    }

  } // nonaggregate

} // dr1467


namespace dr1490 {  // dr1490: yes c++11
  // List-initialization from a string literal

  char s[4]{"abc"};                   // Ok
  std::initializer_list<char>{"abc"}; // expected-error {{expected unqualified-id}}}

} // dr1490

namespace dr1589 {   // dr1589: yes c++11
  // Ambiguous ranking of list-initialization sequences

  void f0(long, int=0);                 // Would makes selection of #0 ambiguous
  void f0(long);                        // #0
  void f0(std::initializer_list<int>);  // #00
  void g0() { f0({1L}); }               // chooses #00
  
  void f1(int, int=0);                    // Would make selection of #1 ambiguous
  void f1(int);                           // #1
  void f1(std::initializer_list<long>);   // #2
  void g1() { f1({42}); }                 // chooses #2

  void f2(std::pair<const char*, const char*>, int = 0); // Would makes selection of #3 ambiguous
  void f2(std::pair<const char*, const char*>); // #3
  void f2(std::initializer_list<std::string>);  // #4
  void g2() { f2({"foo","bar"}); }              // chooses #4
  
  namespace with_error {
    
    void f0(long);                        // #0    expected-note {{candidate function}}
    void f0(std::initializer_list<int>);  // #00   expected-note {{candidate function}}
    void f0(std::initializer_list<int>, int = 0);  // Makes selection of #00 ambiguous \
                                                 // expected-note {{candidate function}}
    void g0() { f0({1L}); }                 // chooses #00    expected-error{{call to 'f0' is ambiguous}}
    
    void f1(int);                           // #1   expected-note {{candidate function}}
    void f1(std::initializer_list<long>);   // #2   expected-note {{candidate function}}
    void f1(std::initializer_list<long>, int = 0);   // Makes selection of #00 ambiguous \
                                                   // expected-note {{candidate function}}
    void g1() { f1({42}); }                 // chooses #2   expected-error{{call to 'f1' is ambiguous}}

    void f2(std::pair<const char*, const char*>); // #3   TODO: expected- note {{candidate function}}
    void f2(std::initializer_list<std::string>);  // #4   expected-note {{candidate function}}
    void f2(std::initializer_list<std::string>, int = 0);   // Makes selection of #00 ambiguous \
                                                          // expected-note {{candidate function}}
    void g2() { f2({"foo","bar"}); }        // chooses #4   expected-error{{call to 'f2' is ambiguous}}

  }

} // dr1589


namespace dr1631 {  // dr1589: yes c++11
  // Incorrect overload resolution for single-element initializer-list

  struct A { int a[1]; };
  struct B { B(int); };
  void f(B, int);
  void f(B, int, int = 0);
  void f(int, A);

  void test() {
    f({0}, {{1}});
  }

  namespace with_error {
    void f(B, int);           // TODO: expected- note {{candidate function}}
    void f(int, A);           // expected-note {{candidate function}}
    void f(int, A, int = 0);  // expected-note {{candidate function}}

    void test() {
      f({0}, {{1}});        // expected-error{{call to 'f' is ambiguous}}
    }
  }

} // dr1631

namespace dr1756 {  // dr1490: yes c++11
  // Direct-list-initialization of a non-class object
  
  int a{0};
  
  struct X { operator int(); } x;
  int b{x};
}

namespace dr1758 {  // dr1758: yes c++11
  // Explicit conversion in copy/move list initialization

  struct X { X(); };
  struct Y { explicit operator X(); } y;
  X x{y};

  struct A {
    A() {}
    A(const A &) {}
  };
  struct B {
    operator A() { return A(); }
  } b;
  A a{b};

} // dr1758

#endif

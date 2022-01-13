// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1413 { // dr1413: 12
  template<int> struct Check {
    typedef int type;
  };
  template<typename T> struct A : T {
    static const int a = 1;
    static const int b;
    static void c();
    void d();

    void f() {
      Check<true ? 0 : A::unknown_spec>::type *var1; // expected-error {{undeclared identifier 'var1'}}
      Check<true ? 0 : a>::type *var2; // ok, variable declaration  expected-note 0+{{here}}
      Check<true ? 0 : b>::type *var3; // expected-error {{undeclared identifier 'var3'}}
      Check<true ? 0 : ((void)c, 0)>::type *var4; // expected-error {{undeclared identifier 'var4'}}
      // value-dependent because of the implied type-dependent 'this->', not because of 'd'
      Check<true ? 0 : (d(), 0)>::type *var5; // expected-error {{undeclared identifier 'var5'}}
      // value-dependent because of the value-dependent '&' operator, not because of 'A::d'
      Check<true ? 0 : (&A::d(), 0)>::type *var5; // expected-error {{undeclared identifier 'var5'}}
    }
  };
}

namespace dr1423 { // dr1423: 11
#if __cplusplus >= 201103L
  bool b1 = nullptr; // expected-error {{cannot initialize}}
  bool b2(nullptr); // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  bool b3 = {nullptr}; // expected-error {{cannot initialize}}
  bool b4{nullptr}; // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
#endif
}

namespace dr1443 { // dr1443: yes
struct A {
  int i;
  A() { void foo(int=i); } // expected-error {{default argument references 'this'}}
};
}

// dr1425: na abi

namespace dr1460 { // dr1460: 3.5
#if __cplusplus >= 201103L
  namespace DRExample {
    union A {
      union {}; // expected-error {{does not declare anything}}
      union {}; // expected-error {{does not declare anything}}
      constexpr A() {}
    };
    constexpr A a = A();

    union B {
      union {}; // expected-error {{does not declare anything}}
      union {}; // expected-error {{does not declare anything}}
      constexpr B() = default;
    };
    constexpr B b = B();

    union C {
      union {}; // expected-error {{does not declare anything}}
      union {}; // expected-error {{does not declare anything}}
    };
    constexpr C c = C();
#if __cplusplus > 201103L
    constexpr void f() { C c; }
    static_assert((f(), true), "");
#endif
  }

  union A {};
  union B { int n; }; // expected-note 0+{{here}}
  union C { int n = 0; };
  struct D { union {}; }; // expected-error {{does not declare anything}}
  struct E { union { int n; }; }; // expected-note 0+{{here}}
  struct F { union { int n = 0; }; };

  struct X {
    friend constexpr A::A() noexcept;
    friend constexpr B::B() noexcept;
#if __cplusplus <= 201703L
    // expected-error@-2 {{follows non-constexpr declaration}}
#endif
    friend constexpr C::C() noexcept;
    friend constexpr D::D() noexcept;
    friend constexpr E::E() noexcept;
#if __cplusplus <= 201703L
    // expected-error@-2 {{follows non-constexpr declaration}}
#endif
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
    union B { int n; constexpr B() = default; };
#if __cplusplus <= 201703L
    // expected-error@-2 {{not constexpr}}
#endif
    union C { int n = 0; constexpr C() = default; };
    struct D { union {}; constexpr D() = default; }; // expected-error {{does not declare anything}}
    struct E { union { int n; }; constexpr E() = default; };
#if __cplusplus <= 201703L
    // expected-error@-2 {{not constexpr}}
#endif
    struct F { union { int n = 0; }; constexpr F() = default; };

    struct G { union { int n = 0; }; union { int m; }; constexpr G() = default; };
#if __cplusplus <= 201703L
    // expected-error@-2 {{not constexpr}}
#endif
    struct H {
      union {
        int n = 0;
      };
      union { // expected-note 0-2{{member not initialized}}
        int m;
      };
      constexpr H() {}
#if __cplusplus <= 201703L
      // expected-error@-2 {{initialize all members}}
#endif
      constexpr H(bool) : m(1) {}
      constexpr H(char) : n(1) {}
#if __cplusplus <= 201703L
      // expected-error@-2 {{initialize all members}}
#endif
      constexpr H(double) : m(1), n(1) {}
    };
  }

#if __cplusplus > 201103L
  template<typename T> constexpr bool check() {
    T t;
#if __cplusplus <= 201703L
    // expected-note-re@-2 2{{non-constexpr constructor '{{[BE]}}'}}
#endif
    return true;
  }
  static_assert(check<A>(), "");
  static_assert(check<B>(), "");
#if __cplusplus <= 201703L
  // expected-error@-2 {{constant}} expected-note@-2 {{in call}}
#endif
  static_assert(check<C>(), "");
  static_assert(check<D>(), "");
  static_assert(check<E>(), "");
#if __cplusplus <= 201703L
  // expected-error@-2 {{constant}} expected-note@-2 {{in call}}
#endif
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
    : __begin_(__b), __size_(__s) {}

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
} // std

namespace dr1467 {  // dr1467: 3.7 c++11
  // Note that the change to [over.best.ics] was partially undone by DR2076;
  // the resulting rule is tested with the tests for that change.

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

    void f1(int);                                  // expected-note {{candidate function}}
    void f1(std::initializer_list<long>) = delete; // expected-note {{candidate function has been explicitly deleted}}
    void g1() { f1({42}); }                        // expected-error {{call to deleted function 'f1'}}

    template <class T, class U>
    struct Pair {
      Pair(T, U);
    };
    struct String {
      String(const char *);
    };

    void f2(Pair<const char *, const char *>);       // expected-note {{candidate function}}
    void f2(std::initializer_list<String>) = delete; // expected-note {{candidate function has been explicitly deleted}}
    void g2() { f2({"foo", "bar"}); }                // expected-error {{call to deleted function 'f2'}}
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

  namespace SelfInitIsNotListInit {
    struct S {
      S();
      explicit S(S &);
      S(const S &);
    };
    S s1;
    S s2 = {s1}; // ok, not list-initialization so we pick the non-explicit constructor
  }

  struct NestedInit { int a, b, c; };
  NestedInit ni[1] = {{NestedInit{1, 2, 3}}};

  namespace NestedInit2 {
    struct Pair { int a, b; };
    struct TwoPairs { TwoPairs(Pair, Pair); };
    struct Value { Value(Pair); Value(TwoPairs); };
    void f() { Value{{{1,2},{3,4}}}; }
  }
  namespace NonAmbiguous {
  // The original implementation made this case ambiguous due to the special
  // handling of one element initialization lists.
  void f(int(&&)[1]);
  void f(unsigned(&&)[1]);

  void g(unsigned i) {
    f({i});
  }
  } // namespace NonAmbiguous

#if __cplusplus >= 201103L
  namespace StringLiterals {
  // When the array size is 4 the call will attempt to bind an lvalue to an
  // rvalue and fail. Therefore #2 will be called. (rsmith will bring this
  // issue to CWG)
  void f(const char(&&)[4]);              // expected-note 2 {{expects an rvalue}} expected-note 3 {{no known conversion}}
  void f(const char(&&)[5]) = delete;     // expected-note 2 {{candidate function has been explicitly deleted}} expected-note 3 {{no known conversion}}
  void f(const wchar_t(&&)[4]);           // expected-note {{expects an rvalue}} expected-note 4 {{no known conversion}}
  void f(const wchar_t(&&)[5]) = delete;  // expected-note {{candidate function has been explicitly deleted}} expected-note 4 {{no known conversion}}
#if __cplusplus >= 202002L
  void f2(const char8_t(&&)[4]);          // expected-note {{expects an rvalue}}
  void f2(const char8_t(&&)[5]) = delete; // expected-note {{candidate function has been explicitly deleted}}
#endif
  void f(const char16_t(&&)[4]);          // expected-note {{expects an rvalue}} expected-note 4 {{no known conversion}}
  void f(const char16_t(&&)[5]) = delete; // expected-note {{candidate function has been explicitly deleted}} expected-note 4 {{no known conversion}}
  void f(const char32_t(&&)[4]);          // expected-note {{expects an rvalue}} expected-note 4 {{no known conversion}}
  void f(const char32_t(&&)[5]) = delete; // expected-note {{candidate function has been explicitly deleted}} expected-note 4 {{no known conversion}}
  void g() {
    f({"abc"});       // expected-error {{call to deleted function 'f'}}
    f({((("abc")))}); // expected-error {{call to deleted function 'f'}}
    f({L"abc"});      // expected-error {{call to deleted function 'f'}}
#if __cplusplus >= 202002L
    f2({u8"abc"});    // expected-error {{call to deleted function 'f2'}}
#endif
    f({uR"(abc)"});   // expected-error {{call to deleted function 'f'}}
    f({(UR"(abc)")}); // expected-error {{call to deleted function 'f'}}
  }
  } // namespace StringLiterals
#endif
} // dr1467

namespace dr1490 {  // dr1490: 3.7 c++11
  // List-initialization from a string literal

  char s[4]{"abc"};                   // Ok
  std::initializer_list<char>{"abc"}; // expected-error {{expected unqualified-id}}}
} // dr190

namespace dr1495 { // dr1495: 4
  // Deduction succeeds in both directions.
  template<typename T, typename U> struct A {}; // expected-note {{template is declared here}}
  template<typename T, typename U> struct A<U, T> {}; // expected-error {{class template partial specialization is not more specialized}}

  // Primary template is more specialized.
  template<typename, typename...> struct B {}; // expected-note {{template is declared here}}
  template<typename ...Ts> struct B<Ts...> {}; // expected-error {{not more specialized}}

  // Deduction fails in both directions.
  template<int, typename, typename ...> struct C {}; // expected-note {{template is declared here}}
  template<typename ...Ts> struct C<0, Ts...> {}; // expected-error {{not more specialized}}

#if __cplusplus >= 201402L
  // Deduction succeeds in both directions.
  template<typename T, typename U> int a; // expected-note {{template is declared here}}
  template<typename T, typename U> int a<U, T>; // expected-error {{variable template partial specialization is not more specialized}}

  // Primary template is more specialized.
  template<typename, typename...> int b; // expected-note {{template is declared here}}
  template<typename ...Ts> int b<Ts...>; // expected-error {{not more specialized}}

  // Deduction fails in both directions.
  template<int, typename, typename ...> int c; // expected-note {{template is declared here}}
  template<typename ...Ts> int c<0, Ts...>; // expected-error {{not more specialized}}
#endif
}
#endif

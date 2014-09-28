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

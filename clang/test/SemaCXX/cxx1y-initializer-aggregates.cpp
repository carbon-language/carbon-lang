// RUN: %clang_cc1 -std=c++1y %s -verify

namespace in_class_init {
  union U { char c; double d = 4.0; };
  constexpr U u1 = U();
  constexpr U u2 {};
  constexpr U u3 { 'x' };
  static_assert(u1.d == 4.0, "");
  static_assert(u2.d == 4.0, "");
  static_assert(u3.c == 'x', "");

  struct A {
    int n = 5;
    int m = n * 3;
    union {
      char c;
      double d = 4.0;
    };
  };
  constexpr A a1 {};
  constexpr A a2 { 8 };
  constexpr A a3 { 1, 2, { 3 } };
  constexpr A a4 { 1, 2, { .d = 3.0 } };
  static_assert(a1.d == 4.0, "");
  static_assert(a2.m == 24, "");
  static_assert(a2.d == 4.0, "");
  static_assert(a3.c == 3, "");
  static_assert(a3.d == 4.0, ""); // expected-error {{constant expression}} expected-note {{active member 'c'}}
  static_assert(a4.d == 3.0, "");

  struct B {
    int n;
    constexpr int f() { return n * 5; }
    int m = f();
  };
  B b1 {};
  constexpr B b2 { 2 };
  B b3 { 1, 2 };
  static_assert(b2.m == 10, "");

  struct C {
    int k;
    union {
      int l = k; // expected-error {{invalid use of non-static}}
    };
  };
}

namespace nested_aggregate_init {
  struct A {
    int n = 5;
    int b = n * 3;
  };
  struct B {
    constexpr B(int k) : d(1.23), k(k) {}
    // Within this aggregate, both this object's 'this' and the temporary's
    // 'this' are used.
    constexpr int f() const { return A{k}.b; }
    double d;
    int k;
  };
  static_assert(B(6).f() == 18, "");
}

namespace use_self {
  struct FibTree {
    int n;
    FibTree *l = // expected-note {{declared here}}
      n > 1 ? new FibTree{n-1} : &fib0; // expected-error {{default member initializer for 'l' needed}}
    FibTree *r = // expected-note {{declared here}}
      n > 2 ? new FibTree{n-2} : &fib0; // expected-error {{default member initializer for 'r' needed}}
    int v = l->v + r->v;

    static FibTree fib0;
  };
  FibTree FibTree::fib0{0, nullptr, nullptr, 1};

  int fib(int n) { return FibTree{n}.v; }
}

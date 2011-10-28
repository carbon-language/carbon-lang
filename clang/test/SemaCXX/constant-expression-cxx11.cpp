// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// FIXME: support const T& parameters here.
//template<typename T> constexpr T id(const T &t) { return t; }
template<typename T> constexpr T id(T t) { return t; }
// FIXME: support templates here.
//template<typename T> constexpr T min(const T &a, const T &b) {
//  return a < b ? a : b;
//}
//template<typename T> constexpr T max(const T &a, const T &b) {
//  return a < b ? b : a;
//}
constexpr int min(int a, int b) { return a < b ? a : b; }
constexpr int max(int a, int b) { return a < b ? b : a; }

struct MemberZero {
  constexpr int zero() { return 0; }
};

namespace TemplateArgumentConversion {
  template<int n> struct IntParam {};

  using IntParam0 = IntParam<0>;
  // FIXME: This should be accepted once we do constexpr function invocation.
  using IntParam0 = IntParam<id(0)>; // expected-error {{not an integral constant expression}}
  using IntParam0 = IntParam<MemberZero().zero>; // expected-error {{did you mean to call it with no arguments?}} expected-error {{not an integral constant expression}}
}

namespace CaseStatements {
  void f(int n) {
    switch (n) {
    // FIXME: Produce the 'add ()' fixit for this.
    case MemberZero().zero: // desired-error {{did you mean to call it with no arguments?}} expected-error {{not an integer constant expression}}
    // FIXME: This should be accepted once we do constexpr function invocation.
    case id(1): // expected-error {{not an integer constant expression}}
      return;
    }
  }
}

extern int &Recurse1;
int &Recurse2 = Recurse1, &Recurse1 = Recurse2;
constexpr int &Recurse3 = Recurse2; // expected-error {{must be initialized by a constant expression}}

namespace MemberEnum {
  struct WithMemberEnum {
    enum E { A = 42 };
  } wme;
  // FIXME: b's initializer is not treated as a constant expression yet, but we
  // can at least fold it.
  constexpr bool b = wme.A == 42;
  int n[b];
}

namespace Recursion {
  constexpr int fib(int n) { return n > 1 ? fib(n-1) + fib(n-2) : n; }
  // FIXME: this isn't an ICE yet.
  using check_fib = int[fib(11)];
  using check_fib = int[89];

  constexpr int gcd_inner(int a, int b) {
    return b == 0 ? a : gcd_inner(b, a % b);
  }
  constexpr int gcd(int a, int b) {
    return gcd_inner(max(a, b), min(a, b));
  }

  // FIXME: this isn't an ICE yet.
  using check_gcd = int[gcd(1749237, 5628959)];
  using check_gcd = int[7];
}

namespace FunctionCast {
  // When folding, we allow functions to be cast to different types. Such
  // cast functions cannot be called, even if they're constexpr.
  constexpr int f() { return 1; }
  typedef double (*DoubleFn)();
  typedef int (*IntFn)();
  int a[(int)DoubleFn(f)()]; // expected-error {{variable length array}}
  int b[(int)IntFn(f)()];    // ok
}

namespace StaticMemberFunction {
  struct S {
    static constexpr int k = 42;
    static constexpr int f(int n) { return n * k + 2; }
  } s;
  // FIXME: this isn't an ICE yet.
  using check_static_call = int[S::f(19)];
  constexpr int n = s.f(19);
  using check_static_call = int[s.f(19)];
  using check_static_call = int[800];
}

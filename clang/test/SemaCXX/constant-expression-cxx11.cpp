// RUN: %clang_cc1 -triple i686-linux -fsyntax-only -verify -std=c++11 %s

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
constexpr int min(const int &a, const int &b) { return a < b ? a : b; }
constexpr int max(const int &a, const int &b) { return a < b ? b : a; }

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

namespace ParameterScopes {

  const int k = 42;
  constexpr const int &ObscureTheTruth(const int &a) { return a; }
  constexpr const int &MaybeReturnJunk(bool b, const int a) {
    return ObscureTheTruth(b ? a : k);
  }
  constexpr int n1 = MaybeReturnJunk(false, 0); // ok
  constexpr int n2 = MaybeReturnJunk(true, 0); // expected-error {{must be initialized by a constant expression}}

  constexpr int InternalReturnJunk(int n) {
    // FIXME: We should reject this: it never produces a constant expression.
    return MaybeReturnJunk(true, n);
  }
  constexpr int n3 = InternalReturnJunk(0); // expected-error {{must be initialized by a constant expression}}

  constexpr int LToR(int &n) { return n; }
  constexpr int GrabCallersArgument(bool which, int a, int b) {
    return LToR(which ? b : a);
  }
  constexpr int n4 = GrabCallersArgument(false, 1, 2);
  constexpr int n5 = GrabCallersArgument(true, 4, 8);
  // FIXME: this isn't an ICE yet.
  using check_value = int[n4 + n5];
  using check_value = int[9];

}

namespace Pointers {

  constexpr int f(int n, const int *a, const int *b, const int *c) {
    return n == 0 ? 0 : *a + f(n-1, b, c, a);
  }

  const int x = 1, y = 10, z = 100;
  constexpr int n1 = f(23, &x, &y, &z);
  // FIXME: this isn't an ICE yet.
  using check_value_1 = int[n1];
  using check_value_1 = int[788];

  constexpr int g(int n, int a, int b, int c) {
    return f(n, &a, &b, &c);
  }
  constexpr int n2 = g(23, x, y, z);
  using check_value_1 = int[n2];

}

namespace FunctionPointers {

  constexpr int Double(int n) { return 2 * n; }
  constexpr int Triple(int n) { return 3 * n; }
  constexpr int Twice(int (*F)(int), int n) { return F(F(n)); }
  constexpr int Quadruple(int n) { return Twice(Double, n); }
  constexpr auto Select(int n) -> int (*)(int) {
    return n == 2 ? &Double : n == 3 ? &Triple : n == 4 ? &Quadruple : 0;
  }
  constexpr int Apply(int (*F)(int), int n) { return F(n); }

  using check_value = int[1 + Apply(Select(4), 5) + Apply(Select(3), 7)];
  using check_value = int[42];

  constexpr int Invalid = Apply(Select(0), 0); // expected-error {{must be initialized by a constant expression}}

}

namespace PointerComparison {

int x, y;
constexpr bool g1 = &x == &y;
constexpr bool g2 = &x != &y;
constexpr bool g3 = &x <= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g4 = &x >= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g5 = &x < &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g6 = &x > &y; // expected-error {{must be initialized by a constant expression}}

struct S { int x, y; } s;
constexpr bool m1 = &s.x == &s.y;
constexpr bool m2 = &s.x != &s.y;
constexpr bool m3 = &s.x <= &s.y;
constexpr bool m4 = &s.x >= &s.y;
constexpr bool m5 = &s.x < &s.y;
constexpr bool m6 = &s.x > &s.y;

constexpr bool n1 = 0 == &y;
constexpr bool n2 = 0 != &y;
constexpr bool n3 = 0 <= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n4 = 0 >= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n5 = 0 < &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n6 = 0 > &y; // expected-error {{must be initialized by a constant expression}}

constexpr bool n7 = &x == 0;
constexpr bool n8 = &x != 0;
constexpr bool n9 = &x <= 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n10 = &x >= 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n11 = &x < 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n12 = &x > 0; // expected-error {{must be initialized by a constant expression}}

constexpr bool s1 = &x == &x;
constexpr bool s2 = &x != &x;
constexpr bool s3 = &x <= &x;
constexpr bool s4 = &x >= &x;
constexpr bool s5 = &x < &x;
constexpr bool s6 = &x > &x;

constexpr S* sptr = &s;
constexpr bool dyncast = sptr == dynamic_cast<S*>(sptr);

using check = int[m1 + (m2<<1) + (m3<<2) + (m4<<3) + (m5<<4) + (m6<<5) +
                  (n1<<6) + (n2<<7) + (n7<<8) + (n8<<9) + (g1<<10) + (g2<<11) +
               (s1<<12) + (s2<<13) + (s3<<14) + (s4<<15) + (s5<<16) + (s6<<17)];
using check = int[2+4+16+128+512+2048+4096+16384+32768];

}

namespace MaterializeTemporary {

constexpr int f(const int &r) { return r; }
constexpr int n = f(1);

constexpr bool same(const int &a, const int &b) { return &a == &b; }
constexpr bool sameTemporary(const int &n) { return same(n, n); }

using check_value = int[1];
using check_value = int[n];
using check_value = int[!same(4, 4)];
using check_value = int[same(n, n)];
using check_value = int[sameTemporary(9)];

}

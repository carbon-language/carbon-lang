// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fcxx-exceptions %s

namespace StdExample {

constexpr int f(void *) { return 0; }
constexpr int f(...) { return 1; }
constexpr int g1() { return f(0); }
constexpr int g2(int n) { return f(n); }
constexpr int g3(int n) { return f(n*0); }

namespace N {
  constexpr int c = 5;
  constexpr int h() { return c; }
}
constexpr int c = 0;
constexpr int g4() { return N::h(); }

static_assert(f(0) == 0, "");
static_assert(f('0') == 1, "");
static_assert(g1() == 0, "");
static_assert(g2(0) == 1, "");
static_assert(g2(1) == 1, "");
static_assert(g3(0) == 1, "");
static_assert(g3(1) == 1, "");
static_assert(N::h() == 5, "");
static_assert(g4() == 5, "");


constexpr int f(bool b)
  { return b ? throw 0 : 0; } // ok
constexpr int f() { return throw 0, 0; } // expected-error {{constexpr function never produces a constant expression}} expected-note {{subexpression}}

struct B {
  constexpr B(int x) : i(0) { }
  int i;
};

int global; // expected-note {{declared here}}

struct D : B {
  constexpr D() : B(global) { } // expected-error {{constexpr constructor never produces a constant expression}} expected-note {{read of non-const}}
};

}

namespace PotentialConstant {

constexpr int Comma(int n) { return // expected-error {{constexpr function never produces a constant expression}}
  (void)(n * 2),
  throw 0, // expected-note {{subexpression}}
  0;
}

int ng; // expected-note 5{{here}}
constexpr int BinaryOp1(int n) { return n + ng; } // expected-error {{never produces}} expected-note {{read}}
constexpr int BinaryOp2(int n) { return ng + n; } // expected-error {{never produces}} expected-note {{read}}

double dg; // expected-note 2{{here}}
constexpr double BinaryOp1(double d) { return d + dg; } // expected-error {{never produces}} expected-note {{read}}
constexpr double BinaryOp2(double d) { return dg + d; } // expected-error {{never produces}} expected-note {{read}}

constexpr int Add(int a, int b, int c) { return a + b + c; }
constexpr int FunctionArgs(int a) { return Add(a, ng, a); } // expected-error {{never produces}} expected-note {{read}}

struct S { int a; int b; int c[2]; };
constexpr S InitList(int a) { return { a, ng }; }; // expected-error {{never produces}} expected-note {{read}}
constexpr S InitList2(int a) { return { a, a, { ng } }; }; // expected-error {{never produces}} expected-note {{read}}

constexpr S InitList3(int a) { return a ? (S){ a, a } : (S){ a, ng }; }; // ok

// FIXME: Check both arms of a ?: if the conditional is a potential constant
// expression with an unknown value, and diagnose if neither is constant.
constexpr S InitList4(int a) { return a ? (S){ a, ng } : (S){ a, ng }; };

}

// Since -fdelayed-template-parsing is enabled by default on Windows (as a
// Microsoft extension), -fno-delayed-template-parsing is used for the tests in
// order to have the same behavior on all systems.
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -pass-by-value %t.cpp -- -std=c++11 -fno-delayed-template-parsing -I %S
// RUN: FileCheck -input-file=%t.cpp %s
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -pass-by-value %t.cpp -- -std=c++11 -fno-delayed-template-parsing -I %S
// RUN: FileCheck -check-prefix=SAFE_RISK -input-file=%t.cpp %s

#include "basic.h"
// CHECK: #include <utility>

// Test that when the class declaration can't be modified we won't modify the
// definition either.
UnmodifiableClass::UnmodifiableClass(const Movable &M) : M(M) {}
// CHECK: UnmodifiableClass::UnmodifiableClass(const Movable &M) : M(M) {}

struct A {
  A(const Movable &M) : M(M) {}
  // CHECK: A(Movable M) : M(std::move(M)) {}
  // SAFE_RISK: A(const Movable &M) : M(M) {}
  Movable M;
};

// Test that we aren't modifying other things than a parameter
Movable GlobalObj;
struct B {
  B(const Movable &M) : M(GlobalObj) {}
  // CHECK: B(const Movable &M) : M(GlobalObj) {}
  Movable M;
};

// Test that a parameter with more than one reference to it won't be changed.
struct C {
  // Tests extra-reference in body
  C(const Movable &M) : M(M) { this->i = M.a; }
  // CHECK: C(const Movable &M) : M(M) { this->i = M.a; }

  // Tests extra-reference in init-list
  C(const Movable &M, int) : M(M), i(M.a) {}
  // CHECK: C(const Movable &M, int) : M(M), i(M.a) {}
  Movable M;
  int i;
};

// Test that both declaration and definition are updated
struct D {
  D(const Movable &M);
  // CHECK: D(Movable M);
  Movable M;
};
D::D(const Movable &M) : M(M) {}
// CHECK: D::D(Movable M) : M(std::move(M)) {}

// Test with default parameter
struct E {
  E(const Movable &M = Movable()) : M(M) {}
  // CHECK: E(Movable M = Movable()) : M(std::move(M)) {}
  Movable M;
};

// Test with object that can't be moved
struct F {
  F(const NotMovable &NM) : NM(NM) {}
  // CHECK: F(const NotMovable &NM) : NM(NM) {}
  NotMovable NM;
};

// Test unnamed parameter in declaration
struct G {
  G(const Movable &);
  // CHECK: G(Movable );
  Movable M;
};
G::G(const Movable &M) : M(M) {}
// CHECK: G::G(Movable M) : M(std::move(M)) {}

// Test parameter with and without qualifier
namespace ns_H {
typedef ::Movable HMovable;
}
struct H {
  H(const ns_H::HMovable &M);
  // CHECK: H(ns_H::HMovable M);
  ns_H::HMovable M;
};
using namespace ns_H;
H::H(const HMovable &M) : M(M) {}
// CHECK: H(HMovable M) : M(std::move(M)) {}

// Try messing up with macros
#define MOVABLE_PARAM(Name) const Movable & Name
// CHECK: #define MOVABLE_PARAM(Name) const Movable & Name
struct I {
  I(MOVABLE_PARAM(M)) : M(M) {}
  // CHECK: I(MOVABLE_PARAM(M)) : M(M) {}
  Movable M;
};
#undef MOVABLE_PARAM

// Test that templates aren't modified
template <typename T> struct J {
  J(const T &M) : M(M) {}
  // CHECK: J(const T &M) : M(M) {}
  T M;
};
J<Movable> j1(Movable());
J<NotMovable> j2(NotMovable());

struct K_Movable {
  K_Movable() = default;
  K_Movable(const K_Movable &) = default;
  K_Movable(K_Movable &&o) { dummy = o.dummy; }
  int dummy;
};

// Test with movable type with an user defined move constructor.
struct K {
  K(const K_Movable &M) : M(M) {}
  // CHECK: K(K_Movable M) : M(std::move(M)) {}
  K_Movable M;
};

template <typename T> struct L {
  L(const Movable &M) : M(M) {}
  // CHECK: L(Movable M) : M(std::move(M)) {}
  Movable M;
};
L<int> l(Movable());

// Test with a non-instantiated template class
template <typename T> struct N {
  N(const Movable &M) : M(M) {}
  // CHECK: N(Movable M) : M(std::move(M)) {}

  Movable M;
  T A;
};

// Test with value parameter
struct O {
  O(Movable M) : M(M) {}
  // CHECK: O(Movable M) : M(std::move(M)) {}
  Movable M;
};

// Test with a const-value parameter
struct P {
  P(const Movable M) : M(M) {}
  // CHECK: P(const Movable M) : M(M) {}
  Movable M;
};

// Test with multiples parameters where some need to be changed and some don't
// need to.
struct Q {
  Q(const Movable &A, const Movable &B, const Movable &C, double D)
      : A(A), B(B), C(C), D(D) {}
  // CHECK:      Q(const Movable &A, Movable B, Movable C, double D)
  // CHECK-NEXT:     : A(A), B(std::move(B)), C(std::move(C)), D(D) {}
  const Movable &A;
  Movable B;
  Movable C;
  double D;
};

// Test that value-parameters with a nested name specifier are left as-is
namespace ns_R {
typedef ::Movable RMovable;
}
struct R {
  R(ns_R::RMovable M) : M(M) {}
  // CHECK: R(ns_R::RMovable M) : M(std::move(M)) {}
  ns_R::RMovable M;
};

// Test with rvalue parameter
struct S {
  S(Movable &&M) : M(M) {}
  // CHECK: S(Movable &&M) : M(M) {}
  Movable M;
};

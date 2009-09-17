// RUN: clang-cc -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct X {};

// CHECK: define void @_Z1f1XS_(
void f(X, X) { }

// CHECK: define void @_Z1fR1XS0_(
void f(X&, X&) { }

// CHECK: define void @_Z1fRK1XS1_(
void f(const X&, const X&) { }

typedef void T();
struct S {};

// CHECK: define void @_Z1fPFvvEM1SFvvE(
void f(T*, T (S::*)) {}

namespace A {
  struct A { };
  struct B { };
};

// CHECK: define void @_Z1fN1A1AENS_1BE(
void f(A::A a, A::B b) { }

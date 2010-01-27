// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Non-trivial dtors, should both be passed indirectly.
struct S {
  ~S();
  int s;
};

// CHECK: define void @_Z1fv(%struct.S* noalias sret %
S f() { return S(); }
// CHECK: define void @_Z1f1S(%struct.S*)
void f(S) { }

// Non-trivial dtors, should both be passed indirectly.
class C {
  ~C();
  double c;
};

// CHECK: define void @_Z1gv(%class.C* noalias sret %
C g() { return C(); }

// CHECK: define void @_Z1f1C(%class.C*) 
void f(C) { }

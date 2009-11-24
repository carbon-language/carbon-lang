// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A;

struct B {
  virtual void f();
  virtual A g();
};

void B::f() { }

// CHECK: declare void @_ZN1B1gEv()

struct C;

struct D {
  virtual void f();
  virtual C g();
};

void D::f() { }

struct C {
  int a;
};

// CHECK: define i64 @_ZN1D1gEv(%struct.B* %this)
C D::g() {
  return C();
}

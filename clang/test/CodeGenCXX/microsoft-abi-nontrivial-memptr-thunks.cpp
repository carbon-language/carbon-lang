// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple=i386-pc-win32 %s -verify

struct A {
  A();
  ~A();
  int a;
};
struct B {
  virtual void f(A); // expected-error {{cannot compile this non-trivial argument copy for thunk yet}}
};
void (B::*mp)(A) = &B::f;

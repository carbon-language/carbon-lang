// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc %s -emit-llvm-only -verify

// We reject this because LLVM doesn't forward the second regparm through the
// thunk.

struct A {
  virtual void __fastcall f(int a, int b); // expected-error {{cannot compile this pointer to fastcall virtual member function yet}}
};
void (__fastcall A::*doit())(int, int) {
  return &A::f;
}

// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace {
  struct A {
    virtual void f() { }
  };
}

void f() { A b; }

struct B {
  B();
  virtual void f();
};

B::B() { }

// B has a key function that is not defined in this translation unit so its vtable
// has external linkage.
// CHECK: @_ZTV1B = external constant

// The A vtable should have internal linkage since it is inside an anonymous 
// namespace.
// CHECK: @_ZTVN12_GLOBAL__N_11AE = internal constant

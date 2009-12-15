// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: _ZTS1B = constant
// CHECK: _ZTS1A = weak_odr constant
// CHECK: _ZTI1A = weak_odr constant
// CHECK: _ZTI1B = constant

// A has no key function, so its RTTI data should be weak_odr.
struct A { };

// B has a key function defined in the translation unit, so the RTTI data should
// be emitted in this translation unit and have external linkage.
struct B : A {
  virtual void f();
};
void B::f() { }

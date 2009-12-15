// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

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

struct C {
  C();
  virtual void f() { } 
};

C::C() { } 

struct D {
  virtual void f();
};

void D::f() { }

static struct : D { } e;

// B has a key function that is not defined in this translation unit so its vtable
// has external linkage.
// CHECK: @_ZTV1B = external constant

// C has no key function, so its vtable should have weak_odr linkage.
// CHECK: @_ZTS1C = weak_odr constant
// CHECK: @_ZTI1C = weak_odr constant
// CHECK: @_ZTV1C = weak_odr constant

// D has a key function that is defined in this translation unit so its vtable is
// defined in the translation unit.
// CHECK: @_ZTS1D = constant
// CHECK: @_ZTI1D = constant
// CHECK: @_ZTV1D = constant

// The anonymous struct for e has no linkage, so the vtable should have
// internal linkage.
// CHECK: @"_ZTS3$_0" = internal constant
// CHECK: @"_ZTI3$_0" = internal constant
// CHECK: @"_ZTV3$_0" = internal constant

// The A vtable should have internal linkage since it is inside an anonymous 
// namespace.
// CHECK: @_ZTSN12_GLOBAL__N_11AE = internal constant
// CHECK: @_ZTIN12_GLOBAL__N_11AE = internal constant
// CHECK: @_ZTVN12_GLOBAL__N_11AE = internal constant

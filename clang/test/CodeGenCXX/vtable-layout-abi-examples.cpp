// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm-only -fdump-vtable-layouts 2>&1 | FileCheck %s

/// Examples from the Itanium C++ ABI specification.
/// http://www.codesourcery.com/public/cxx-abi/

namespace Test1 {
  
// This is from http://www.codesourcery.com/public/cxx-abi/cxx-vtable-ex.html

// CHECK:      Vtable for 'Test1::A' (5 entries).
// CHECK-NEXT:    0 | offset_to_top (0)
// CHECK-NEXT:    1 | Test1::A RTTI
// CHECK-NEXT:        -- (Test1::A, 0) vtable address --
// CHECK-NEXT:    2 | void Test1::A::f()
// CHECK-NEXT:    3 | void Test1::A::g()
// CHECK-NEXT:    4 | void Test1::A::h()
struct A {
  virtual void f ();
  virtual void g ();
  virtual void h ();
  int ia;
};
void A::f() {}
  
}

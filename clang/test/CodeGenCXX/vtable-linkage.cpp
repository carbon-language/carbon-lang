// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace {
  // The vtables should have internal linkage.
  struct A {
    virtual void f() { }
  };
  
  struct B : virtual A {
    virtual void f() { } 
  };

  // CHECK: @_ZTVN12_GLOBAL__N_11BE = internal constant
  // CHECK: @_ZTTN12_GLOBAL__N_11BE = internal constant
  // CHECK: @_ZTVN12_GLOBAL__N_11AE = internal constant
}

void f() { B b; }

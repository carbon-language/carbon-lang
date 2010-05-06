// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

int f();

namespace {
  // CHECK: @_ZN12_GLOBAL__N_11bE = internal global i32 0
  // CHECK: @_ZN12_GLOBAL__N_1L1cE = internal global i32 0
  // CHECK: @_ZN12_GLOBAL__N_11D1dE = internal global i32 0
  // CHECK: @_ZN12_GLOBAL__N_11aE = internal global i32 0
  int a = 0;

  int b = f();

  static int c = f();

  class D {
    static int d;
  };
  
  int D::d = f();

  // Check for generation of a VTT with internal linkage
  // CHECK: @_ZTSN12_GLOBAL__N_11X1EE = internal constant
  struct X { 
    struct EBase { };
    struct E : public virtual EBase { virtual ~E() {} };
  };

  // CHECK: define internal i32 @_ZN12_GLOBAL__N_13fooEv()
  int foo() {
    return 32;
  }

  // CHECK: define internal i32 @_ZN12_GLOBAL__N_11A3fooEv()
  namespace A {
    int foo() {
      return 45;
    }
  }
}

int concrete() {
  return a + foo() + A::foo();
}

void test_XE() { throw X::E(); }

// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

namespace {
  // CHECK: @_ZN12_GLOBAL__N_11aE = internal global i32 0
  int a = 0;

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

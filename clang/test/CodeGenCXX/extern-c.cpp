// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
namespace foo {

// CHECK-NOT: @a = global i32
extern "C" int a;

// CHECK-NOT: @_ZN3foo1bE = global i32
extern int b;

// CHECK: @_ZN3foo1cE = global i32
int c = 5;

// CHECK-NOT: @_ZN3foo1dE
extern "C" struct d;

}

namespace test1 {
  namespace {
    struct X {};
  }
  extern "C" {
    // CHECK: @b = global
    X b = X();
  }
  void *use = &b;
  // CHECK: @_ZN5test13useE = global
}

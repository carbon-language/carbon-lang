// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// rdar://problem/14985269.
//
// Verify that the global init helper function does not get associated
// with any source location.
//
// CHECK: define internal void @_GLOBAL__I_a
// CHECK-NOT: !dbg
// CHECK: "_GLOBAL__I_a", i32 0, {{.*}}, i32 0} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
# 99 "someheader.h"
class A {
public:
  A();
  int foo() { return 0; }
};
# 5 "main.cpp"
A a;

int f() {
  return a.foo();
}


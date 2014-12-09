// RUN: %clang_cc1 -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

int &src(); int* sink();

void f1() {
#line 100
  * // The store for the assignment should be attributed to the start of the
    // assignment expression here, regardless of the location of subexpressions.
  sink() = src();
  // CHECK: store {{.*}}, !dbg [[DBG_F1:!.*]]
}

struct foo {
  int i;
  int &j;
  foo();
};

foo::foo()
  :
#line 200
    i
    (src()),
    j
    (src())
    // CHECK: store i32 {{.*}} !dbg [[DBG_FOO_VALUE:!.*]]
    // CHECK: store i32* {{.*}} !dbg [[DBG_FOO_REF:!.*]]
{
}

// CHECK: [[DBG_F1]] = metadata !{i32 100,
// CHECK: [[DBG_FOO_VALUE]] = metadata !{i32 200,
// CHECK: [[DBG_FOO_REF]] = metadata !{i32 202,

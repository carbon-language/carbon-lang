// RUN: %clang_cc1 -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

int &src();
int* sink();
__complex float complex_src();

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
  __complex float k;
  foo();
};

foo::foo()
  :
#line 200
    i
    (src()),
    // CHECK: store i32 {{.*}} !dbg [[DBG_FOO_VALUE:!.*]]
    j
    (src()),
    // CHECK: store i32* {{.*}} !dbg [[DBG_FOO_REF:!.*]]
    k
    (complex_src())
    // CHECK: store float {{.*}} !dbg [[DBG_FOO_COMPLEX:!.*]]
{
}

// CHECK: [[DBG_F1]] = metadata !{i32 100,
// CHECK: [[DBG_FOO_VALUE]] = metadata !{i32 200,
// CHECK: [[DBG_FOO_REF]] = metadata !{i32 203,
// CHECK: [[DBG_FOO_COMPLEX]] = metadata !{i32 206,

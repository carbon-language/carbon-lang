// RUN: %clang_cc1 -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

int &src();
int *sink();
__complex float complex_src();
__complex float *complex_sink();

// CHECK-LABEL: define
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

// CHECK-LABEL: define
foo::foo()
    :
#line 200
      i // CHECK: store i32 {{.*}} !dbg [[DBG_FOO_VALUE:!.*]]
      (src()),
      j // CHECK: store i32* {{.*}} !dbg [[DBG_FOO_REF:!.*]]
      (src()),
      k // CHECK: store float {{.*}} !dbg [[DBG_FOO_COMPLEX:!.*]]
      (complex_src()) {
}

// skip C1
// CHECK-LABEL: define

// CHECK-LABEL: define
void f2() {
#line 300
  * // CHECK: store float {{.*}} !dbg [[DBG_F2:!.*]]
      complex_sink() = complex_src();
}

// CHECK-LABEL: define
void f3() {
#line 400
  * // CHECK: store float {{.*}} !dbg [[DBG_F3:!.*]]
      complex_sink() += complex_src();
}

// CHECK: [[DBG_F1]] = metadata !{i32 100,
// CHECK: [[DBG_FOO_VALUE]] = metadata !{i32 200,
// CHECK: [[DBG_FOO_REF]] = metadata !{i32 202,
// CHECK: [[DBG_FOO_COMPLEX]] = metadata !{i32 204,
// CHECK: [[DBG_F2]] = metadata !{i32 300,
// CHECK: [[DBG_F3]] = metadata !{i32 400,

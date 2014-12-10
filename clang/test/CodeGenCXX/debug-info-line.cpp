// RUN: %clang_cc1 -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

int &src();
int *sink();
extern "C" __complex float complex_src();
extern "C" __complex float *complex_sink();

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

// CHECK-LABEL: define {{.*}}f2{{.*}}
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

// CHECK-LABEL: define
void f4() {
#line 500
  auto x // CHECK: store {{.*}} !dbg [[DBG_F4:!.*]]
      = src();
}

// CHECK-LABEL: define
void f5() {
#line 600
  auto x // CHECK: store float {{.*}} !dbg [[DBG_F5:!.*]]
      = complex_src();
}

struct agg { int i; };
agg agg_src();

// CHECK-LABEL: define
void f6() {
  agg x;
#line 700
  x // CHECK: call void @llvm.memcpy{{.*}} !dbg [[DBG_F6:!.*]]
      = agg_src();
}

// CHECK-LABEL: define
void f7() {
  int *src1();
  int src2();
#line 800
  int x = ( // CHECK: load {{.*}} !dbg [[DBG_F7:!.*]]
      src1())[src2()];
}

// CHECK-LABEL: define
void f8() {
  int src1[1];
  int src2();
#line 900
  int x = ( // CHECK: load {{.*}} !dbg [[DBG_F8:!.*]]
      src1)[src2()];
}

// CHECK: [[DBG_F1]] = metadata !{i32 100,
// CHECK: [[DBG_FOO_VALUE]] = metadata !{i32 200,
// CHECK: [[DBG_FOO_REF]] = metadata !{i32 202,
// CHECK: [[DBG_FOO_COMPLEX]] = metadata !{i32 204,
// CHECK: [[DBG_F2]] = metadata !{i32 300,
// CHECK: [[DBG_F3]] = metadata !{i32 400,
// CHECK: [[DBG_F4]] = metadata !{i32 500,
// CHECK: [[DBG_F5]] = metadata !{i32 600,
// CHECK: [[DBG_F6]] = metadata !{i32 700,
// CHECK: [[DBG_F7]] = metadata !{i32 800,
// CHECK: [[DBG_F8]] = metadata !{i32 900,

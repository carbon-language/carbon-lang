// RUN: %clang_cc1 -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s

int src(); int* sink();

void f1() {
#line 100
  * // The store for the assignment should be attributed to the start of the
    // assignment expression here, regardless of the location of subexpressions.
  (
  sink
  (
  )
  +
  3
  )
  =
  src
  (
  )
  +
  42
  ;
  // CHECK: store {{.*}}, !dbg [[DBG1:!.*]]
}

// CHECK: [[DBG1]] = metadata !{i32 100, {{.*}}

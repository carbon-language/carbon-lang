// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -emit-llvm %s -o - -verify | FileCheck %s
// expected-no-diagnostics

// CHECK: @foo = external global float
extern constant float foo;

kernel void test(global float* buf) {
  buf[0] += foo;
}

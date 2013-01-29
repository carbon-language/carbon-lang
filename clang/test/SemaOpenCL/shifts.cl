// RUN: %clang_cc1 -x cl -O0 -emit-llvm  %s -o - -triple x86_64-linux-gnu | FileCheck %s
// OpenCL essentially reduces all shift amounts to the last word-size bits before evaluating.
// Test this both for variables and constants evaluated in the front-end.

// CHECK: @gtest1 = constant i64 2147483648
__constant const unsigned long gtest1 = 1UL << 31;

// CHECK: @negativeShift32
int negativeShift32(int a,int b) {
  // CHECK: %array0 = alloca [256 x i8]
  char array0[((int)1)<<40];
  // CHECK: %array1 = alloca [256 x i8]
  char array1[((int)1)<<(-24)];

  // CHECK: ret i32 65536
  return ((int)1)<<(-16);
}

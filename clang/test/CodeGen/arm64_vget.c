// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD vget intrinsics

#include <arm_neon.h>

float64_t test_vget_lane_f64(float64x1_t a1) {
  // CHECK: test_vget_lane_f64
  // why isn't 1 allowed as second argument?
  return vget_lane_f64(a1, 0);
  // CHECK: extractelement {{.*}} i32 0
  // CHECK-NEXT: ret
}


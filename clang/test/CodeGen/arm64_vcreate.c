// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD vcreate intrinsics

/*#include <arm_neon.h>*/
#include <arm_neon.h>

float32x2_t test_vcreate_f32(uint64_t a1) {
  // CHECK: test_vcreate_f32
  return vcreate_f32(a1);
  // CHECK: bitcast {{.*}} to <2 x float>
  // CHECK-NEXT: ret
}

// FIXME enable when scalar_to_vector in backend is fixed.  Also, change
// CHECK@ to CHECK<colon> and CHECK-NEXT@ to CHECK-NEXT<colon>
/*
float64x1_t test_vcreate_f64(uint64_t a1) {
  // CHECK@ test_vcreate_f64
  return vcreate_f64(a1);
  // CHECK@ llvm.aarch64.neon.saddlv.i64.v2i32
  // CHECK-NEXT@ ret
}
*/

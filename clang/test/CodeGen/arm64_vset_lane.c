// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD set lane intrinsics INCOMPLETE

#include <arm_neon.h>

float16x4_t test_vset_lane_f16(float16_t *a1, float16x4_t a2) {
  // CHECK-LABEL: test_vset_lane_f16
  return vset_lane_f16(*a1, a2, 1);
  // CHECK insertelement <4 x i16> %a2, i16 %a1, i32 1
}

float16x8_t test_vsetq_lane_f16(float16_t *a1, float16x8_t a2) {
  // CHECK-LABEL: test_vsetq_lane_f16
  return vsetq_lane_f16(*a1, a2, 4);
  // CHECK insertelement <8 x i16> %a2, i16 %a1, i32 4
}

// problem with scalar_to_vector in backend.  Punt for now
#if 0
float64x1_t test_vset_lane_f64(float64_t a1, float64x1_t a2) {
  // CHECK-LABEL@ test_vset_lane_f64
  return vset_lane_f64(a1, a2, 0);
  // CHECK@ @llvm.arm64.neon.smaxv.i32.v8i8
}
#endif

float64x2_t test_vsetq_lane_f64(float64_t a1, float64x2_t a2) {
  // CHECK-LABEL: test_vsetq_lane_f64
  return vsetq_lane_f64(a1, a2, 0);
  // CHECK insertelement <2 x double> %a2, double %a1, i32 0
}

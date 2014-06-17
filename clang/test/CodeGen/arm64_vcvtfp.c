// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s

#include <arm_neon.h>

float64x2_t test_vcvt_f64_f32(float32x2_t x) {
  // CHECK-LABEL: test_vcvt_f64_f32
  return vcvt_f64_f32(x);
  // CHECK: fpext <2 x float> {{%.*}} to <2 x double>
  // CHECK-NEXT: ret
}

float64x2_t test_vcvt_high_f64_f32(float32x4_t x) {
  // CHECK-LABEL: test_vcvt_high_f64_f32
  return vcvt_high_f64_f32(x);
  // CHECK: [[HIGH:%.*]] = shufflevector <4 x float> {{%.*}}, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  // CHECK-NEXT: fpext <2 x float> [[HIGH]] to <2 x double>
  // CHECK-NEXT: ret
}

float32x2_t test_vcvt_f32_f64(float64x2_t v) {
  // CHECK: test_vcvt_f32_f64
  return vcvt_f32_f64(v);
  // CHECK: fptrunc <2 x double> {{%.*}} to <2 x float>
  // CHECK-NEXT: ret
}

float32x4_t test_vcvt_high_f32_f64(float32x2_t x, float64x2_t v) {
  // CHECK: test_vcvt_high_f32_f64
  return vcvt_high_f32_f64(x, v);
  // CHECK: [[TRUNC:%.*]] = fptrunc <2 x double> {{.*}} to <2 x float>
  // CHECK-NEXT: shufflevector <2 x float> {{.*}}, <2 x float> [[TRUNC]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK-NEXT: ret
}

float32x2_t test_vcvtx_f32_f64(float64x2_t v) {
  // CHECK: test_vcvtx_f32_f64
  return vcvtx_f32_f64(v);
  // CHECK: llvm.aarch64.neon.fcvtxn.v2f32.v2f64
  // CHECK-NEXT: ret
}

float32x4_t test_vcvtx_high_f32_f64(float32x2_t x, float64x2_t v) {
  // CHECK: test_vcvtx_high_f32_f64
  return vcvtx_high_f32_f64(x, v);
  // CHECK: llvm.aarch64.neon.fcvtxn.v2f32.v2f64
  // CHECK: shufflevector
  // CHECK: ret
}

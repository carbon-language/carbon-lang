// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>


float32_t test_vmuls_lane_f32(float32_t a, float32x2_t b) {
  // CHECK: test_vmuls_lane_f32
  return vmuls_lane_f32(a, b, 1);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float64_t test_vmuld_lane_f64(float64_t a, float64x1_t b) {
  // CHECK: test_vmuld_lane_f64
  return vmuld_lane_f64(a, b, 0);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float32_t test_vmuls_laneq_f32(float32_t a, float32x4_t b) {
  // CHECK: test_vmuls_laneq_f32
  return vmuls_laneq_f32(a, b, 3);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmuld_laneq_f64(float64_t a, float64x2_t b) {
  // CHECK: test_vmuld_laneq_f64
  return vmuld_laneq_f64(a, b, 1);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

float64x1_t test_vmul_n_f64(float64x1_t a, float64_t b) {
  // CHECK: test_vmul_n_f64
  return vmul_n_f64(a, b);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float32_t test_vmulxs_lane_f32(float32_t a, float32x2_t b) {
// CHECK: test_vmulxs_lane_f32
  return vmulxs_lane_f32(a, b, 1);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float32_t test_vmulxs_laneq_f32(float32_t a, float32x4_t b) {
// CHECK: test_vmulxs_laneq_f32
  return vmulxs_laneq_f32(a, b, 3);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmulxd_lane_f64(float64_t a, float64x1_t b) {
// CHECK: test_vmulxd_lane_f64
  return vmulxd_lane_f64(a, b, 0);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float64_t test_vmulxd_laneq_f64(float64_t a, float64x2_t b) {
// CHECK: test_vmulxd_laneq_f64
  return vmulxd_laneq_f64(a, b, 1);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK_AARCH64: test_vmulx_lane_f64
float64x1_t test_vmulx_lane_f64(float64x1_t a, float64x1_t b) {
  return vmulx_lane_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}


// CHECK_AARCH64: test_vmulx_laneq_f64_0
float64x1_t test_vmulx_laneq_f64_0(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK_AARCH64: test_vmulx_laneq_f64_1
float64x1_t test_vmulx_laneq_f64_1(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 1);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}


// CHECK_AARCH64: test_vfmas_lane_f32
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
  // CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK_AARCH64: test_vfmad_lane_f64
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK_AARCH64: test_vfmad_laneq_f64
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK_AARCH64: test_vfmss_lane_f32
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmss_lane_f32(a, b, c, 1);
  // CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK_AARCH64: test_vfma_lane_f64
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK_AARCH64: test_vfms_lane_f64
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfms_lane_f64(a, b, v, 0);
  // CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK_AARCH64: test_vfma_laneq_f64
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK_AARCH64: test_vfms_laneq_f64
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfms_laneq_f64(a, b, v, 0);
  // CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}


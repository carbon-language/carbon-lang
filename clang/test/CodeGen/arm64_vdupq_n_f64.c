// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | \
// RUN:   FileCheck -check-prefix=CHECK-IR %s
// REQUIRES: arm64-registered-target

/// Test vdupq_n_f64 and vmovq_nf64 ARM64 intrinsics
// <rdar://problem/11778405> ARM64: vdupq_n_f64 and vdupq_lane_f64 intrinsics
// missing


#include <arm_neon.h>

// vdupq_n_f64 -> dup.2d v0, v0[0]
//
float64x2_t test_vdupq_n_f64(float64_t w)
{
    return vdupq_n_f64(w);
  // CHECK-LABEL: test_vdupq_n_f64:
  // CHECK: dup.2d v0, v0[0]
  // CHECK-NEXT: ret
}

// might as well test this while we're here
// vdupq_n_f32 -> dup.4s v0, v0[0]
float32x4_t test_vdupq_n_f32(float32_t w)
{
    return vdupq_n_f32(w);
  // CHECK-LABEL: test_vdupq_n_f32:
  // CHECK: dup.4s v0, v0[0]
  // CHECK-NEXT: ret
}

// vdupq_lane_f64 -> dup.2d v0, v0[0]
// this was in <rdar://problem/11778405>, but had already been implemented,
// test anyway
float64x2_t test_vdupq_lane_f64(float64x1_t V)
{
    return vdupq_lane_f64(V, 0);
  // CHECK-LABEL: test_vdupq_lane_f64:
  // CHECK: dup.2d v0, v0[0]
  // CHECK-NEXT: ret
}

// vmovq_n_f64 -> dup Vd.2d,X0
// this wasn't in <rdar://problem/11778405>, but it was between the vdups
float64x2_t test_vmovq_n_f64(float64_t w)
{
  return vmovq_n_f64(w);
  // CHECK-LABEL: test_vmovq_n_f64:
  // CHECK: dup.2d v0, v0[0]
  // CHECK-NEXT: ret
}

float16x4_t test_vmov_n_f16(float16_t *a1)
{
  // CHECK-IR-LABEL: test_vmov_n_f16
  return vmov_n_f16(*a1);
  // CHECK-IR: insertelement {{.*}} i32 0{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 1{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 2{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 3{{ *$}}
}

// Disable until scalar problem in backend is fixed. Change CHECK-IR@ to
// CHECK-IR<colon>
/*
float64x1_t test_vmov_n_f64(float64_t a1)
{
  // CHECK-IR@ test_vmov_n_f64
  return vmov_n_f64(a1);
  // CHECK-IR@ insertelement {{.*}} i32 0{{ *$}}
}
*/

float16x8_t test_vmovq_n_f16(float16_t *a1)
{
  // CHECK-IR-LABEL: test_vmovq_n_f16
  return vmovq_n_f16(*a1);
  // CHECK-IR: insertelement {{.*}} i32 0{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 1{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 2{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 3{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 4{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 5{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 6{{ *$}}
  // CHECK-IR: insertelement {{.*}} i32 7{{ *$}}
}


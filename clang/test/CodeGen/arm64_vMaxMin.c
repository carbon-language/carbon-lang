// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - %s | FileCheck -check-prefix=CHECK-CODEGEN %s
// REQUIRES: arm64-registered-target
// Test ARM64 SIMD max/min intrinsics

#include <arm_neon.h>

// Test a represntative sample of 8 and 16, signed and unsigned, 64 and 128 bit reduction
int8_t test_vmaxv_s8(int8x8_t a1) {
  // CHECK: test_vmaxv_s8
  return vmaxv_s8(a1);
  // CHECK @llvm.arm64.neon.smaxv.i32.v8i8
}

uint16_t test_vminvq_u16(uint16x8_t a1) {
  // CHECK: test_vminvq_u16
  return vminvq_u16(a1);
  // CHECK llvm.arm64.neon.uminv.i16.v8i16
}

// Test a represntative sample of 8 and 16, signed and unsigned, 64 and 128 bit pairwise
uint8x8_t test_vmin_u8(uint8x8_t a1, uint8x8_t a2) {
  // CHECK: test_vmin_u8
  return vmin_u8(a1, a2);
  // CHECK llvm.arm64.neon.umin.v8i8
}

uint8x16_t test_vminq_u8(uint8x16_t a1, uint8x16_t a2) {
  // CHECK: test_vminq_u8
  return vminq_u8(a1, a2);
  // CHECK llvm.arm64.neon.umin.v16i8
}

int16x8_t test_vmaxq_s16(int16x8_t a1, int16x8_t a2) {
  // CHECK: test_vmaxq_s16
  return vmaxq_s16(a1, a2);
  // CHECK llvm.arm64.neon.smax.v8i16
}

// Test the more complicated cases of [suf]32 and f64
float64x2_t test_vmaxq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vmaxq_f64
  return vmaxq_f64(a1, a2);
  // CHECK llvm.arm64.neon.fmax.v2f64
}

float32x4_t test_vmaxq_f32(float32x4_t a1, float32x4_t a2) {
  // CHECK: test_vmaxq_f32
  return vmaxq_f32(a1, a2);
  // CHECK llvm.arm64.neon.fmax.v4f32
}

float64x2_t test_vminq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vminq_f64
  return vminq_f64(a1, a2);
  // CHECK llvm.arm64.neon.fmin.v2f64
}

float32x2_t test_vmax_f32(float32x2_t a1, float32x2_t a2) {
  // CHECK: test_vmax_f32
  return vmax_f32(a1, a2);
  // CHECK llvm.arm64.neon.fmax.v2f32
}

int32x2_t test_vmax_s32(int32x2_t a1, int32x2_t a2) {
  // CHECK: test_vmax_s32
  return vmax_s32(a1, a2);
  // CHECK llvm.arm64.neon.smax.v2i32
}

uint32x2_t test_vmin_u32(uint32x2_t a1, uint32x2_t a2) {
  // CHECK: test_vmin_u32
  return vmin_u32(a1, a2);
  // CHECK llvm.arm64.neon.umin.v2i32
}

float32_t test_vmaxnmv_f32(float32x2_t a1) {
  // CHECK: test_vmaxnmv_f32
  return vmaxnmv_f32(a1);
  // CHECK: llvm.arm64.neon.fmaxnmv.f32.v2f32
  // CHECK-NEXT: ret
}

// this doesn't translate into a valid instruction, regardless of what the
// ARM doc says.
#if 0
float64_t test_vmaxnmvq_f64(float64x2_t a1) {
  // CHECK@ test_vmaxnmvq_f64
  return vmaxnmvq_f64(a1);
  // CHECK@ llvm.arm64.neon.saddlv.i64.v2i32
  // CHECK-NEXT@ ret
}
#endif

float32_t test_vmaxnmvq_f32(float32x4_t a1) {
  // CHECK: test_vmaxnmvq_f32
  return vmaxnmvq_f32(a1);
  // CHECK: llvm.arm64.neon.fmaxnmv.f32.v4f32
  // CHECK-NEXT: ret
}

float32_t test_vmaxv_f32(float32x2_t a1) {
  // CHECK: test_vmaxv_f32
  return vmaxv_f32(a1);
  // CHECK: llvm.arm64.neon.fmaxv.f32.v2f32
  // FIXME check that the 2nd and 3rd arguments are the same V register below
  // CHECK-CODEGEN: fmaxp.2s
  // CHECK-NEXT: ret
}

int32_t test_vmaxv_s32(int32x2_t a1) {
  // CHECK: test_vmaxv_s32
  return vmaxv_s32(a1);
  // CHECK: llvm.arm64.neon.smaxv.i32.v2i32
  // FIXME check that the 2nd and 3rd arguments are the same V register below
  // CHECK-CODEGEN: smaxp.2s
  // CHECK-NEXT: ret
}

uint32_t test_vmaxv_u32(uint32x2_t a1) {
  // CHECK: test_vmaxv_u32
  return vmaxv_u32(a1);
  // CHECK: llvm.arm64.neon.umaxv.i32.v2i32
  // FIXME check that the 2nd and 3rd arguments are the same V register below
  // CHECK-CODEGEN: umaxp.2s
  // CHECK-NEXT: ret
}

// FIXME punt on this for now; don't forget to fix CHECKs
#if 0
float64_t test_vmaxvq_f64(float64x2_t a1) {
  // CHECK@ test_vmaxvq_f64
  return vmaxvq_f64(a1);
  // CHECK@ llvm.arm64.neon.fmaxv.i64.v2f64
  // CHECK-NEXT@ ret
}
#endif

float32_t test_vmaxvq_f32(float32x4_t a1) {
  // CHECK: test_vmaxvq_f32
  return vmaxvq_f32(a1);
  // CHECK: llvm.arm64.neon.fmaxv.f32.v4f32
  // CHECK-NEXT: ret
}

float32_t test_vminnmv_f32(float32x2_t a1) {
  // CHECK: test_vminnmv_f32
  return vminnmv_f32(a1);
  // CHECK: llvm.arm64.neon.fminnmv.f32.v2f32
  // CHECK-NEXT: ret
}

float32_t test_vminvq_f32(float32x4_t a1) {
  // CHECK: test_vminvq_f32
  return vminvq_f32(a1);
  // CHECK: llvm.arm64.neon.fminv.f32.v4f32
  // CHECK-NEXT: ret
}

// this doesn't translate into a valid instruction, regardless of what the ARM
// doc says.
#if 0
float64_t test_vminnmvq_f64(float64x2_t a1) {
  // CHECK@ test_vminnmvq_f64
  return vminnmvq_f64(a1);
  // CHECK@ llvm.arm64.neon.saddlv.i64.v2i32
  // CHECK-NEXT@ ret
}
#endif

float32_t test_vminnmvq_f32(float32x4_t a1) {
  // CHECK: test_vminnmvq_f32
  return vminnmvq_f32(a1);
  // CHECK: llvm.arm64.neon.fminnmv.f32.v4f32
  // CHECK-NEXT: ret
}

float32_t test_vminv_f32(float32x2_t a1) {
  // CHECK: test_vminv_f32
  return vminv_f32(a1);
  // CHECK: llvm.arm64.neon.fminv.f32.v2f32
  // CHECK-NEXT: ret
}

int32_t test_vminv_s32(int32x2_t a1) {
  // CHECK: test_vminv_s32
  return vminv_s32(a1);
  // CHECK: llvm.arm64.neon.sminv.i32.v2i32
  // CHECK-CODEGEN: sminp.2s
  // CHECK-NEXT: ret
}

uint32_t test_vminv_u32(uint32x2_t a1) {
  // CHECK: test_vminv_u32
  return vminv_u32(a1);
  // CHECK: llvm.arm64.neon.fminv.f32.v2f32
}

// FIXME punt on this for now; don't forget to fix CHECKs
#if 0
float64_t test_vminvq_f64(float64x2_t a1) {
  // CHECK@ test_vminvq_f64
  return vminvq_f64(a1);
  // CHECK@ llvm.arm64.neon.saddlv.i64.v2i32
  // CHECK-NEXT@ ret
}
#endif

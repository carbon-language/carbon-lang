// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s

// Test ARM64 SIMD copy vector element to vector element: vcopyq_lane*

#include <arm_neon.h>

int8x16_t test_vcopyq_laneq_s8(int8x16_t a1, int8x16_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_s8
  return vcopyq_laneq_s8(a1, (int64_t) 3, a2, (int64_t) 13);
  // CHECK: shufflevector <16 x i8> %a1, <16 x i8> %a2, <16 x i32> <i32 0, i32 1, i32 2, i32 29, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
}

uint8x16_t test_vcopyq_laneq_u8(uint8x16_t a1, uint8x16_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_u8
  return vcopyq_laneq_u8(a1, (int64_t) 3, a2, (int64_t) 13);
  // CHECK: shufflevector <16 x i8> %a1, <16 x i8> %a2, <16 x i32> <i32 0, i32 1, i32 2, i32 29, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

}

int16x8_t test_vcopyq_laneq_s16(int16x8_t a1, int16x8_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_s16
  return vcopyq_laneq_s16(a1, (int64_t) 3, a2, (int64_t) 7);
  // CHECK: shufflevector <8 x i16> %a1, <8 x i16> %a2, <8 x i32> <i32 0, i32 1, i32 2, i32 15, i32 4, i32 5, i32 6, i32 7>

}

uint16x8_t test_vcopyq_laneq_u16(uint16x8_t a1, uint16x8_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_u16
  return vcopyq_laneq_u16(a1, (int64_t) 3, a2, (int64_t) 7);
  // CHECK: shufflevector <8 x i16> %a1, <8 x i16> %a2, <8 x i32> <i32 0, i32 1, i32 2, i32 15, i32 4, i32 5, i32 6, i32 7>

}

int32x4_t test_vcopyq_laneq_s32(int32x4_t a1, int32x4_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_s32
  return vcopyq_laneq_s32(a1, (int64_t) 3, a2, (int64_t) 3);
  // CHECK: shufflevector <4 x i32> %a1, <4 x i32> %a2, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
}

uint32x4_t test_vcopyq_laneq_u32(uint32x4_t a1, uint32x4_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_u32
  return vcopyq_laneq_u32(a1, (int64_t) 3, a2, (int64_t) 3);
  // CHECK: shufflevector <4 x i32> %a1, <4 x i32> %a2, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
}

int64x2_t test_vcopyq_laneq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_s64
  return vcopyq_laneq_s64(a1, (int64_t) 0, a2, (int64_t) 1);
  // CHECK: shufflevector <2 x i64> %a1, <2 x i64> %a2, <2 x i32> <i32 3, i32 1>
}

uint64x2_t test_vcopyq_laneq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_u64
  return vcopyq_laneq_u64(a1, (int64_t) 0, a2, (int64_t) 1);
  // CHECK: shufflevector <2 x i64> %a1, <2 x i64> %a2, <2 x i32> <i32 3, i32 1>
}

float32x4_t test_vcopyq_laneq_f32(float32x4_t a1, float32x4_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_f32
  return vcopyq_laneq_f32(a1, 0, a2, 3);
  // CHECK: shufflevector <4 x float> %a1, <4 x float> %a2, <4 x i32> <i32 7, i32 1, i32 2, i32 3>
}

float64x2_t test_vcopyq_laneq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK-LABEL: test_vcopyq_laneq_f64
  return vcopyq_laneq_f64(a1, 0, a2, 1);
  // CHECK: shufflevector <2 x double> %a1, <2 x double> %a2, <2 x i32> <i32 3, i32 1>
}


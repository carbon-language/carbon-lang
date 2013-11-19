// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

#include <arm_neon.h>

uint8x16_t test_vld1q_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_u8
  return vld1q_dup_u8(a);
  // CHECK: ld1r {v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

uint16x8_t test_vld1q_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_u16
  return vld1q_dup_u16(a);
  // CHECK: ld1r {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

uint32x4_t test_vld1q_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_u32
  return vld1q_dup_u32(a);
  // CHECK: ld1r {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

uint64x2_t test_vld1q_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_u64
  return vld1q_dup_u64(a);
  // CHECK: ld1r {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

int8x16_t test_vld1q_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_s8
  return vld1q_dup_s8(a);
  // CHECK: ld1r {v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

int16x8_t test_vld1q_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_s16
  return vld1q_dup_s16(a);
  // CHECK: ld1r {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

int32x4_t test_vld1q_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_s32
  return vld1q_dup_s32(a);
  // CHECK: ld1r {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

int64x2_t test_vld1q_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_s64
  return vld1q_dup_s64(a);
  // CHECK: ld1r {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

float16x8_t test_vld1q_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_f16
  return vld1q_dup_f16(a);
  // CHECK: ld1r {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

float32x4_t test_vld1q_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_f32
  return vld1q_dup_f32(a);
  // CHECK: ld1r {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

float64x2_t test_vld1q_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_f64
  return vld1q_dup_f64(a);
  // CHECK: ld1r {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

poly8x16_t test_vld1q_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_p8
  return vld1q_dup_p8(a);
  // CHECK: ld1r {v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

poly16x8_t test_vld1q_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_p16
  return vld1q_dup_p16(a);
  // CHECK: ld1r {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

poly64x2_t test_vld1q_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld1q_dup_p64
  return vld1q_dup_p64(a);
  // CHECK: ld1r {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

uint8x8_t test_vld1_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld1_dup_u8
  return vld1_dup_u8(a);
  // CHECK: ld1r {v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

uint16x4_t test_vld1_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld1_dup_u16
  return vld1_dup_u16(a);
  // CHECK: ld1r {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

uint32x2_t test_vld1_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld1_dup_u32
  return vld1_dup_u32(a);
  // CHECK: ld1r {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

uint64x1_t test_vld1_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld1_dup_u64
  return vld1_dup_u64(a);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

int8x8_t test_vld1_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld1_dup_s8
  return vld1_dup_s8(a);
  // CHECK: ld1r {v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

int16x4_t test_vld1_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld1_dup_s16
  return vld1_dup_s16(a);
  // CHECK: ld1r {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

int32x2_t test_vld1_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld1_dup_s32
  return vld1_dup_s32(a);
  // CHECK: ld1r {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

int64x1_t test_vld1_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld1_dup_s64
  return vld1_dup_s64(a);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

float16x4_t test_vld1_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld1_dup_f16
  return vld1_dup_f16(a);
  // CHECK: ld1r {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

float32x2_t test_vld1_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld1_dup_f32
  return vld1_dup_f32(a);
  // CHECK: ld1r {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

float64x1_t test_vld1_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld1_dup_f64
  return vld1_dup_f64(a);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

poly8x8_t test_vld1_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld1_dup_p8
  return vld1_dup_p8(a);
  // CHECK: ld1r {v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

poly16x4_t test_vld1_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld1_dup_p16
  return vld1_dup_p16(a);
  // CHECK: ld1r {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

poly64x1_t test_vld1_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld1_dup_p64
  return vld1_dup_p64(a);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

uint8x16x2_t test_vld2q_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_u8
  return vld2q_dup_u8(a);
  // CHECK: ld2r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

uint16x8x2_t test_vld2q_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_u16
  return vld2q_dup_u16(a);
  // CHECK: ld2r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

uint32x4x2_t test_vld2q_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_u32
  return vld2q_dup_u32(a);
  // CHECK: ld2r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

uint64x2x2_t test_vld2q_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_u64
  return vld2q_dup_u64(a);
  // CHECK: ld2r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

int8x16x2_t test_vld2q_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_s8
  return vld2q_dup_s8(a);
  // CHECK: ld2r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

int16x8x2_t test_vld2q_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_s16
  return vld2q_dup_s16(a);
  // CHECK: ld2r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

int32x4x2_t test_vld2q_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_s32
  return vld2q_dup_s32(a);
  // CHECK: ld2r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

int64x2x2_t test_vld2q_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_s64
  return vld2q_dup_s64(a);
  // CHECK: ld2r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

float16x8x2_t test_vld2q_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_f16
  return vld2q_dup_f16(a);
  // CHECK: ld2r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

float32x4x2_t test_vld2q_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_f32
  return vld2q_dup_f32(a);
  // CHECK: ld2r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

float64x2x2_t test_vld2q_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_f64
  return vld2q_dup_f64(a);
  // CHECK: ld2r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

poly8x16x2_t test_vld2q_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_p8
  return vld2q_dup_p8(a);
  // CHECK: ld2r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

poly16x8x2_t test_vld2q_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_p16
  return vld2q_dup_p16(a);
  // CHECK: ld2r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

poly64x2x2_t test_vld2q_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld2q_dup_p64
  return vld2q_dup_p64(a);
  // CHECK: ld2r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

uint8x8x2_t test_vld2_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld2_dup_u8
  return vld2_dup_u8(a);
  // CHECK: ld2r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

uint16x4x2_t test_vld2_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld2_dup_u16
  return vld2_dup_u16(a);
  // CHECK: ld2r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

uint32x2x2_t test_vld2_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld2_dup_u32
  return vld2_dup_u32(a);
  // CHECK: ld2r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

uint64x1x2_t test_vld2_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld2_dup_u64
  return vld2_dup_u64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

int8x8x2_t test_vld2_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld2_dup_s8
  return vld2_dup_s8(a);
  // CHECK: ld2r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

int16x4x2_t test_vld2_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld2_dup_s16
  return vld2_dup_s16(a);
  // CHECK: ld2r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

int32x2x2_t test_vld2_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld2_dup_s32
  return vld2_dup_s32(a);
  // CHECK: ld2r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

int64x1x2_t test_vld2_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld2_dup_s64
  return vld2_dup_s64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

float16x4x2_t test_vld2_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld2_dup_f16
  return vld2_dup_f16(a);
  // CHECK: ld2r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

float32x2x2_t test_vld2_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld2_dup_f32
  return vld2_dup_f32(a);
  // CHECK: ld2r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

float64x1x2_t test_vld2_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld2_dup_f64
  return vld2_dup_f64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

poly8x8x2_t test_vld2_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld2_dup_p8
  return vld2_dup_p8(a);
  // CHECK: ld2r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

poly16x4x2_t test_vld2_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld2_dup_p16
  return vld2_dup_p16(a);
  // CHECK: ld2r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

poly64x1x2_t test_vld2_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld2_dup_p64
  return vld2_dup_p64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

uint8x16x3_t test_vld3q_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_u8
  return vld3q_dup_u8(a);
  // CHECK: ld3r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b},
  // [{{x[0-9]+|sp}}]
}

uint16x8x3_t test_vld3q_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_u16
  return vld3q_dup_u16(a);
  // CHECK: ld3r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
  // [{{x[0-9]+|sp}}]
}

uint32x4x3_t test_vld3q_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_u32
  return vld3q_dup_u32(a);
  // CHECK: ld3r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
  // [{{x[0-9]+|sp}}]
}

uint64x2x3_t test_vld3q_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_u64
  return vld3q_dup_u64(a);
  // CHECK: ld3r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
  // [{{x[0-9]+|sp}}]
}

int8x16x3_t test_vld3q_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_s8
  return vld3q_dup_s8(a);
  // CHECK: ld3r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b},
  // [{{x[0-9]+|sp}}]
}

int16x8x3_t test_vld3q_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_s16
  return vld3q_dup_s16(a);
  // CHECK: ld3r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
  // [{{x[0-9]+|sp}}]
}

int32x4x3_t test_vld3q_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_s32
  return vld3q_dup_s32(a);
  // CHECK: ld3r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
  // [{{x[0-9]+|sp}}]
}

int64x2x3_t test_vld3q_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_s64
  return vld3q_dup_s64(a);
  // CHECK: ld3r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
  // [{{x[0-9]+|sp}}]
}

float16x8x3_t test_vld3q_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_f16
  return vld3q_dup_f16(a);
  // CHECK: ld3r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
  // [{{x[0-9]+|sp}}]
}

float32x4x3_t test_vld3q_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_f32
  return vld3q_dup_f32(a);
  // CHECK: ld3r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s},
  // [{{x[0-9]+|sp}}]
}

float64x2x3_t test_vld3q_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_f64
  return vld3q_dup_f64(a);
  // CHECK: ld3r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
  // [{{x[0-9]+|sp}}]
}

poly8x16x3_t test_vld3q_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_p8
  return vld3q_dup_p8(a);
  // CHECK: ld3r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b},
  // [{{x[0-9]+|sp}}]
}

poly16x8x3_t test_vld3q_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_p16
  return vld3q_dup_p16(a);
  // CHECK: ld3r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h},
  // [{{x[0-9]+|sp}}]
}

poly64x2x3_t test_vld3q_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld3q_dup_p64
  return vld3q_dup_p64(a);
  // CHECK: ld3r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d},
  // [{{x[0-9]+|sp}}]
}

uint8x8x3_t test_vld3_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld3_dup_u8
  return vld3_dup_u8(a);
  // CHECK: ld3r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b},
  // [{{x[0-9]+|sp}}]
}

uint16x4x3_t test_vld3_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld3_dup_u16
  return vld3_dup_u16(a);
  // CHECK: ld3r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
  // [{{x[0-9]+|sp}}]
}

uint32x2x3_t test_vld3_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld3_dup_u32
  return vld3_dup_u32(a);
  // CHECK: ld3r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
  // [{{x[0-9]+|sp}}]
}

uint64x1x3_t test_vld3_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld3_dup_u64
  return vld3_dup_u64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
  // [{{x[0-9]+|sp}}]
}

int8x8x3_t test_vld3_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld3_dup_s8
  return vld3_dup_s8(a);
  // CHECK: ld3r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b},
  // [{{x[0-9]+|sp}}]
}

int16x4x3_t test_vld3_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld3_dup_s16
  return vld3_dup_s16(a);
  // CHECK: ld3r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
  // [{{x[0-9]+|sp}}]
}

int32x2x3_t test_vld3_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld3_dup_s32
  return vld3_dup_s32(a);
  // CHECK: ld3r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
  // [{{x[0-9]+|sp}}]
}

int64x1x3_t test_vld3_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld3_dup_s64
  return vld3_dup_s64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
  // [{{x[0-9]+|sp}}]
}

float16x4x3_t test_vld3_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld3_dup_f16
  return vld3_dup_f16(a);
  // CHECK: ld3r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
  // [{{x[0-9]+|sp}}]
}

float32x2x3_t test_vld3_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld3_dup_f32
  return vld3_dup_f32(a);
  // CHECK: ld3r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s},
  // [{{x[0-9]+|sp}}]
}

float64x1x3_t test_vld3_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld3_dup_f64
  return vld3_dup_f64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
  // [{{x[0-9]+|sp}}]
}

poly8x8x3_t test_vld3_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld3_dup_p8
  return vld3_dup_p8(a);
  // CHECK: ld3r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b},
  // [{{x[0-9]+|sp}}]
}

poly16x4x3_t test_vld3_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld3_dup_p16
  return vld3_dup_p16(a);
  // CHECK: ld3r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h},
  // [{{x[0-9]+|sp}}]
}

poly64x1x3_t test_vld3_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld3_dup_p64
  return vld3_dup_p64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d},
  // [{{x[0-9]+|sp}}]
}

uint8x16x4_t test_vld4q_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_u8
  return vld4q_dup_u8(a);
  // CHECK: ld4r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b,
  // v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

uint16x8x4_t test_vld4q_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_u16
  return vld4q_dup_u16(a);
  // CHECK: ld4r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
  // v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

uint32x4x4_t test_vld4q_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_u32
  return vld4q_dup_u32(a);
  // CHECK: ld4r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
  // v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

uint64x2x4_t test_vld4q_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_u64
  return vld4q_dup_u64(a);
  // CHECK: ld4r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
  // v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

int8x16x4_t test_vld4q_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_s8
  return vld4q_dup_s8(a);
  // CHECK: ld4r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b,
  // v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

int16x8x4_t test_vld4q_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_s16
  return vld4q_dup_s16(a);
  // CHECK: ld4r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
  // v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

int32x4x4_t test_vld4q_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_s32
  return vld4q_dup_s32(a);
  // CHECK: ld4r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
  // v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

int64x2x4_t test_vld4q_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_s64
  return vld4q_dup_s64(a);
  // CHECK: ld4r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
  // v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

float16x8x4_t test_vld4q_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_f16
  return vld4q_dup_f16(a);
  // CHECK: ld4r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
  // v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

float32x4x4_t test_vld4q_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_f32
  return vld4q_dup_f32(a);
  // CHECK: ld4r {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s,
  // v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

float64x2x4_t test_vld4q_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_f64
  return vld4q_dup_f64(a);
  // CHECK: ld4r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
  // v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
}

poly8x16x4_t test_vld4q_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_p8
  return vld4q_dup_p8(a);
  // CHECK: ld4r {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b,
  // v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
}

poly16x8x4_t test_vld4q_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_p16
  return vld4q_dup_p16(a);
  // CHECK: ld4r {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h,
  // v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
}

poly64x2x4_t test_vld4q_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld4q_dup_p64
  return vld4q_dup_p64(a);
  // CHECK: ld4r {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d,
  // v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
}

uint8x8x4_t test_vld4_dup_u8(uint8_t const *a) {
  // CHECK-LABEL: test_vld4_dup_u8
  return vld4_dup_u8(a);
  // CHECK: ld4r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b,
  // v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

uint16x4x4_t test_vld4_dup_u16(uint16_t const *a) {
  // CHECK-LABEL: test_vld4_dup_u16
  return vld4_dup_u16(a);
  // CHECK: ld4r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
  // v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

uint32x2x4_t test_vld4_dup_u32(uint32_t const *a) {
  // CHECK-LABEL: test_vld4_dup_u32
  return vld4_dup_u32(a);
  // CHECK: ld4r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
  // v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

uint64x1x4_t test_vld4_dup_u64(uint64_t const *a) {
  // CHECK-LABEL: test_vld4_dup_u64
  return vld4_dup_u64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
  // v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

int8x8x4_t test_vld4_dup_s8(int8_t const *a) {
  // CHECK-LABEL: test_vld4_dup_s8
  return vld4_dup_s8(a);
  // CHECK: ld4r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b,
  // v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

int16x4x4_t test_vld4_dup_s16(int16_t const *a) {
  // CHECK-LABEL: test_vld4_dup_s16
  return vld4_dup_s16(a);
  // CHECK: ld4r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
  // v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

int32x2x4_t test_vld4_dup_s32(int32_t const *a) {
  // CHECK-LABEL: test_vld4_dup_s32
  return vld4_dup_s32(a);
  // CHECK: ld4r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
  // v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

int64x1x4_t test_vld4_dup_s64(int64_t const *a) {
  // CHECK-LABEL: test_vld4_dup_s64
  return vld4_dup_s64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
  // v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

float16x4x4_t test_vld4_dup_f16(float16_t const *a) {
  // CHECK-LABEL: test_vld4_dup_f16
  return vld4_dup_f16(a);
  // CHECK: ld4r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
  // v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

float32x2x4_t test_vld4_dup_f32(float32_t const *a) {
  // CHECK-LABEL: test_vld4_dup_f32
  return vld4_dup_f32(a);
  // CHECK: ld4r {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s,
  // v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
}

float64x1x4_t test_vld4_dup_f64(float64_t const *a) {
  // CHECK-LABEL: test_vld4_dup_f64
  return vld4_dup_f64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
  // v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

poly8x8x4_t test_vld4_dup_p8(poly8_t const *a) {
  // CHECK-LABEL: test_vld4_dup_p8
  return vld4_dup_p8(a);
  // CHECK: ld4r {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b,
  // v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
}

poly16x4x4_t test_vld4_dup_p16(poly16_t const *a) {
  // CHECK-LABEL: test_vld4_dup_p16
  return vld4_dup_p16(a);
  // CHECK: ld4r {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h,
  // v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
}

poly64x1x4_t test_vld4_dup_p64(poly64_t const *a) {
  // CHECK-LABEL: test_vld4_dup_p64
  return vld4_dup_p64(a);
  // CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d,
  // v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

uint8x16_t test_vld1q_lane_u8(uint8_t const *a, uint8x16_t b) {
  // CHECK-LABEL: test_vld1q_lane_u8
  return vld1q_lane_u8(a, b, 15);
  // CHECK: ld1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

uint16x8_t test_vld1q_lane_u16(uint16_t const *a, uint16x8_t b) {
  // CHECK-LABEL: test_vld1q_lane_u16
  return vld1q_lane_u16(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

uint32x4_t test_vld1q_lane_u32(uint32_t const *a, uint32x4_t b) {
  // CHECK-LABEL: test_vld1q_lane_u32
  return vld1q_lane_u32(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

uint64x2_t test_vld1q_lane_u64(uint64_t const *a, uint64x2_t b) {
  // CHECK-LABEL: test_vld1q_lane_u64
  return vld1q_lane_u64(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

int8x16_t test_vld1q_lane_s8(int8_t const *a, int8x16_t b) {
  // CHECK-LABEL: test_vld1q_lane_s8
  return vld1q_lane_s8(a, b, 15);
  // CHECK: ld1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

int16x8_t test_vld1q_lane_s16(int16_t const *a, int16x8_t b) {
  // CHECK-LABEL: test_vld1q_lane_s16
  return vld1q_lane_s16(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

int32x4_t test_vld1q_lane_s32(int32_t const *a, int32x4_t b) {
  // CHECK-LABEL: test_vld1q_lane_s32
  return vld1q_lane_s32(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

int64x2_t test_vld1q_lane_s64(int64_t const *a, int64x2_t b) {
  // CHECK-LABEL: test_vld1q_lane_s64
  return vld1q_lane_s64(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

float16x8_t test_vld1q_lane_f16(float16_t const *a, float16x8_t b) {
  // CHECK-LABEL: test_vld1q_lane_f16
  return vld1q_lane_f16(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

float32x4_t test_vld1q_lane_f32(float32_t const *a, float32x4_t b) {
  // CHECK-LABEL: test_vld1q_lane_f32
  return vld1q_lane_f32(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

float64x2_t test_vld1q_lane_f64(float64_t const *a, float64x2_t b) {
  // CHECK-LABEL: test_vld1q_lane_f64
  return vld1q_lane_f64(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

poly8x16_t test_vld1q_lane_p8(poly8_t const *a, poly8x16_t b) {
  // CHECK-LABEL: test_vld1q_lane_p8
  return vld1q_lane_p8(a, b, 15);
  // CHECK: ld1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

poly16x8_t test_vld1q_lane_p16(poly16_t const *a, poly16x8_t b) {
  // CHECK-LABEL: test_vld1q_lane_p16
  return vld1q_lane_p16(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

poly64x2_t test_vld1q_lane_p64(poly64_t const *a, poly64x2_t b) {
  // CHECK-LABEL: test_vld1q_lane_p64
  return vld1q_lane_p64(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

uint8x8_t test_vld1_lane_u8(uint8_t const *a, uint8x8_t b) {
  // CHECK-LABEL: test_vld1_lane_u8
  return vld1_lane_u8(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

uint16x4_t test_vld1_lane_u16(uint16_t const *a, uint16x4_t b) {
  // CHECK-LABEL: test_vld1_lane_u16
  return vld1_lane_u16(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

uint32x2_t test_vld1_lane_u32(uint32_t const *a, uint32x2_t b) {
  // CHECK-LABEL: test_vld1_lane_u32
  return vld1_lane_u32(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

uint64x1_t test_vld1_lane_u64(uint64_t const *a, uint64x1_t b) {
  // CHECK-LABEL: test_vld1_lane_u64
  return vld1_lane_u64(a, b, 0);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

int8x8_t test_vld1_lane_s8(int8_t const *a, int8x8_t b) {
  // CHECK-LABEL: test_vld1_lane_s8
  return vld1_lane_s8(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

int16x4_t test_vld1_lane_s16(int16_t const *a, int16x4_t b) {
  // CHECK-LABEL: test_vld1_lane_s16
  return vld1_lane_s16(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

int32x2_t test_vld1_lane_s32(int32_t const *a, int32x2_t b) {
  // CHECK-LABEL: test_vld1_lane_s32
  return vld1_lane_s32(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

int64x1_t test_vld1_lane_s64(int64_t const *a, int64x1_t b) {
  // CHECK-LABEL: test_vld1_lane_s64
  return vld1_lane_s64(a, b, 0);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

float16x4_t test_vld1_lane_f16(float16_t const *a, float16x4_t b) {
  // CHECK-LABEL: test_vld1_lane_f16
  return vld1_lane_f16(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

float32x2_t test_vld1_lane_f32(float32_t const *a, float32x2_t b) {
  // CHECK-LABEL: test_vld1_lane_f32
  return vld1_lane_f32(a, b, 1);
  // CHECK: ld1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

float64x1_t test_vld1_lane_f64(float64_t const *a, float64x1_t b) {
  // CHECK-LABEL: test_vld1_lane_f64
  return vld1_lane_f64(a, b, 0);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

poly8x8_t test_vld1_lane_p8(poly8_t const *a, poly8x8_t b) {
  // CHECK-LABEL: test_vld1_lane_p8
  return vld1_lane_p8(a, b, 7);
  // CHECK: ld1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

poly16x4_t test_vld1_lane_p16(poly16_t const *a, poly16x4_t b) {
  // CHECK-LABEL: test_vld1_lane_p16
  return vld1_lane_p16(a, b, 3);
  // CHECK: ld1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

poly64x1_t test_vld1_lane_p64(poly64_t const *a, poly64x1_t b) {
  // CHECK-LABEL: test_vld1_lane_p64
  return vld1_lane_p64(a, b, 0);
  // CHECK: ld1r {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
}

uint16x8x2_t test_vld2q_lane_u16(uint16_t const *a, uint16x8x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_u16
  return vld2q_lane_u16(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

uint32x4x2_t test_vld2q_lane_u32(uint32_t const *a, uint32x4x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_u32
  return vld2q_lane_u32(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

uint64x2x2_t test_vld2q_lane_u64(uint64_t const *a, uint64x2x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_u64
  return vld2q_lane_u64(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

int16x8x2_t test_vld2q_lane_s16(int16_t const *a, int16x8x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_s16
  return vld2q_lane_s16(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

int32x4x2_t test_vld2q_lane_s32(int32_t const *a, int32x4x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_s32
  return vld2q_lane_s32(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

int64x2x2_t test_vld2q_lane_s64(int64_t const *a, int64x2x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_s64
  return vld2q_lane_s64(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

float16x8x2_t test_vld2q_lane_f16(float16_t const *a, float16x8x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_f16
  return vld2q_lane_f16(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

float32x4x2_t test_vld2q_lane_f32(float32_t const *a, float32x4x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_f32
  return vld2q_lane_f32(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

float64x2x2_t test_vld2q_lane_f64(float64_t const *a, float64x2x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_f64
  return vld2q_lane_f64(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

poly16x8x2_t test_vld2q_lane_p16(poly16_t const *a, poly16x8x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_p16
  return vld2q_lane_p16(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

poly64x2x2_t test_vld2q_lane_p64(poly64_t const *a, poly64x2x2_t b) {
  // CHECK-LABEL: test_vld2q_lane_p64
  return vld2q_lane_p64(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

uint8x8x2_t test_vld2_lane_u8(uint8_t const *a, uint8x8x2_t b) {
  // CHECK-LABEL: test_vld2_lane_u8
  return vld2_lane_u8(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

uint16x4x2_t test_vld2_lane_u16(uint16_t const *a, uint16x4x2_t b) {
  // CHECK-LABEL: test_vld2_lane_u16
  return vld2_lane_u16(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

uint32x2x2_t test_vld2_lane_u32(uint32_t const *a, uint32x2x2_t b) {
  // CHECK-LABEL: test_vld2_lane_u32
  return vld2_lane_u32(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

uint64x1x2_t test_vld2_lane_u64(uint64_t const *a, uint64x1x2_t b) {
  // CHECK-LABEL: test_vld2_lane_u64
  return vld2_lane_u64(a, b, 0);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

int8x8x2_t test_vld2_lane_s8(int8_t const *a, int8x8x2_t b) {
  // CHECK-LABEL: test_vld2_lane_s8
  return vld2_lane_s8(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

int16x4x2_t test_vld2_lane_s16(int16_t const *a, int16x4x2_t b) {
  // CHECK-LABEL: test_vld2_lane_s16
  return vld2_lane_s16(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

int32x2x2_t test_vld2_lane_s32(int32_t const *a, int32x2x2_t b) {
  // CHECK-LABEL: test_vld2_lane_s32
  return vld2_lane_s32(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

int64x1x2_t test_vld2_lane_s64(int64_t const *a, int64x1x2_t b) {
  // CHECK-LABEL: test_vld2_lane_s64
  return vld2_lane_s64(a, b, 0);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

float16x4x2_t test_vld2_lane_f16(float16_t const *a, float16x4x2_t b) {
  // CHECK-LABEL: test_vld2_lane_f16
  return vld2_lane_f16(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

float32x2x2_t test_vld2_lane_f32(float32_t const *a, float32x2x2_t b) {
  // CHECK-LABEL: test_vld2_lane_f32
  return vld2_lane_f32(a, b, 1);
  // CHECK: ld2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

float64x1x2_t test_vld2_lane_f64(float64_t const *a, float64x1x2_t b) {
  // CHECK-LABEL: test_vld2_lane_f64
  return vld2_lane_f64(a, b, 0);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

poly8x8x2_t test_vld2_lane_p8(poly8_t const *a, poly8x8x2_t b) {
  // CHECK-LABEL: test_vld2_lane_p8
  return vld2_lane_p8(a, b, 7);
  // CHECK: ld2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

poly16x4x2_t test_vld2_lane_p16(poly16_t const *a, poly16x4x2_t b) {
  // CHECK-LABEL: test_vld2_lane_p16
  return vld2_lane_p16(a, b, 3);
  // CHECK: ld2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

poly64x1x2_t test_vld2_lane_p64(poly64_t const *a, poly64x1x2_t b) {
  // CHECK-LABEL: test_vld2_lane_p64
  return vld2_lane_p64(a, b, 0);
  // CHECK: ld2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

uint16x8x3_t test_vld3q_lane_u16(uint16_t const *a, uint16x8x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_u16
  return vld3q_lane_u16(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

uint32x4x3_t test_vld3q_lane_u32(uint32_t const *a, uint32x4x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_u32
  return vld3q_lane_u32(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

uint64x2x3_t test_vld3q_lane_u64(uint64_t const *a, uint64x2x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_u64
  return vld3q_lane_u64(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

int16x8x3_t test_vld3q_lane_s16(int16_t const *a, int16x8x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_s16
  return vld3q_lane_s16(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

int32x4x3_t test_vld3q_lane_s32(int32_t const *a, int32x4x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_s32
  return vld3q_lane_s32(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

int64x2x3_t test_vld3q_lane_s64(int64_t const *a, int64x2x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_s64
  return vld3q_lane_s64(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

float16x8x3_t test_vld3q_lane_f16(float16_t const *a, float16x8x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_f16
  return vld3q_lane_f16(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

float32x4x3_t test_vld3q_lane_f32(float32_t const *a, float32x4x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_f32
  return vld3q_lane_f32(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

float64x2x3_t test_vld3q_lane_f64(float64_t const *a, float64x2x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_f64
  return vld3q_lane_f64(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

poly8x16x3_t test_vld3q_lane_p8(poly8_t const *a, poly8x16x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_p8
  return vld3q_lane_p8(a, b, 15);
  // CHECK: ld3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

poly16x8x3_t test_vld3q_lane_p16(poly16_t const *a, poly16x8x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_p16
  return vld3q_lane_p16(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

poly64x2x3_t test_vld3q_lane_p64(poly64_t const *a, poly64x2x3_t b) {
  // CHECK-LABEL: test_vld3q_lane_p64
  return vld3q_lane_p64(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

uint8x8x3_t test_vld3_lane_u8(uint8_t const *a, uint8x8x3_t b) {
  // CHECK-LABEL: test_vld3_lane_u8
  return vld3_lane_u8(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

uint16x4x3_t test_vld3_lane_u16(uint16_t const *a, uint16x4x3_t b) {
  // CHECK-LABEL: test_vld3_lane_u16
  return vld3_lane_u16(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

uint32x2x3_t test_vld3_lane_u32(uint32_t const *a, uint32x2x3_t b) {
  // CHECK-LABEL: test_vld3_lane_u32
  return vld3_lane_u32(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

uint64x1x3_t test_vld3_lane_u64(uint64_t const *a, uint64x1x3_t b) {
  // CHECK-LABEL: test_vld3_lane_u64
  return vld3_lane_u64(a, b, 0);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

int8x8x3_t test_vld3_lane_s8(int8_t const *a, int8x8x3_t b) {
  // CHECK-LABEL: test_vld3_lane_s8
  return vld3_lane_s8(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

int16x4x3_t test_vld3_lane_s16(int16_t const *a, int16x4x3_t b) {
  // CHECK-LABEL: test_vld3_lane_s16
  return vld3_lane_s16(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

int32x2x3_t test_vld3_lane_s32(int32_t const *a, int32x2x3_t b) {
  // CHECK-LABEL: test_vld3_lane_s32
  return vld3_lane_s32(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

int64x1x3_t test_vld3_lane_s64(int64_t const *a, int64x1x3_t b) {
  // CHECK-LABEL: test_vld3_lane_s64
  return vld3_lane_s64(a, b, 0);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

float16x4x3_t test_vld3_lane_f16(float16_t const *a, float16x4x3_t b) {
  // CHECK-LABEL: test_vld3_lane_f16
  return vld3_lane_f16(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

float32x2x3_t test_vld3_lane_f32(float32_t const *a, float32x2x3_t b) {
  // CHECK-LABEL: test_vld3_lane_f32
  return vld3_lane_f32(a, b, 1);
  // CHECK: ld3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

float64x1x3_t test_vld3_lane_f64(float64_t const *a, float64x1x3_t b) {
  // CHECK-LABEL: test_vld3_lane_f64
  return vld3_lane_f64(a, b, 0);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

poly8x8x3_t test_vld3_lane_p8(poly8_t const *a, poly8x8x3_t b) {
  // CHECK-LABEL: test_vld3_lane_p8
  return vld3_lane_p8(a, b, 7);
  // CHECK: ld3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

poly16x4x3_t test_vld3_lane_p16(poly16_t const *a, poly16x4x3_t b) {
  // CHECK-LABEL: test_vld3_lane_p16
  return vld3_lane_p16(a, b, 3);
  // CHECK: ld3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

poly64x1x3_t test_vld3_lane_p64(poly64_t const *a, poly64x1x3_t b) {
  // CHECK-LABEL: test_vld3_lane_p64
  return vld3_lane_p64(a, b, 0);
  // CHECK: ld3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

uint8x16x4_t test_vld4q_lane_u8(uint8_t const *a, uint8x16x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_u8
  return vld4q_lane_u8(a, b, 15);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

uint16x8x4_t test_vld4q_lane_u16(uint16_t const *a, uint16x8x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_u16
  return vld4q_lane_u16(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

uint32x4x4_t test_vld4q_lane_u32(uint32_t const *a, uint32x4x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_u32
  return vld4q_lane_u32(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

uint64x2x4_t test_vld4q_lane_u64(uint64_t const *a, uint64x2x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_u64
  return vld4q_lane_u64(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

int8x16x4_t test_vld4q_lane_s8(int8_t const *a, int8x16x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_s8
  return vld4q_lane_s8(a, b, 15);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

int16x8x4_t test_vld4q_lane_s16(int16_t const *a, int16x8x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_s16
  return vld4q_lane_s16(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

int32x4x4_t test_vld4q_lane_s32(int32_t const *a, int32x4x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_s32
  return vld4q_lane_s32(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

int64x2x4_t test_vld4q_lane_s64(int64_t const *a, int64x2x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_s64
  return vld4q_lane_s64(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

float16x8x4_t test_vld4q_lane_f16(float16_t const *a, float16x8x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_f16
  return vld4q_lane_f16(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

float32x4x4_t test_vld4q_lane_f32(float32_t const *a, float32x4x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_f32
  return vld4q_lane_f32(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

float64x2x4_t test_vld4q_lane_f64(float64_t const *a, float64x2x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_f64
  return vld4q_lane_f64(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

poly8x16x4_t test_vld4q_lane_p8(poly8_t const *a, poly8x16x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_p8
  return vld4q_lane_p8(a, b, 15);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

poly16x8x4_t test_vld4q_lane_p16(poly16_t const *a, poly16x8x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_p16
  return vld4q_lane_p16(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

poly64x2x4_t test_vld4q_lane_p64(poly64_t const *a, poly64x2x4_t b) {
  // CHECK-LABEL: test_vld4q_lane_p64
  return vld4q_lane_p64(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

uint8x8x4_t test_vld4_lane_u8(uint8_t const *a, uint8x8x4_t b) {
  // CHECK-LABEL: test_vld4_lane_u8
  return vld4_lane_u8(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

uint16x4x4_t test_vld4_lane_u16(uint16_t const *a, uint16x4x4_t b) {
  // CHECK-LABEL: test_vld4_lane_u16
  return vld4_lane_u16(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

uint32x2x4_t test_vld4_lane_u32(uint32_t const *a, uint32x2x4_t b) {
  // CHECK-LABEL: test_vld4_lane_u32
  return vld4_lane_u32(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

uint64x1x4_t test_vld4_lane_u64(uint64_t const *a, uint64x1x4_t b) {
  // CHECK-LABEL: test_vld4_lane_u64
  return vld4_lane_u64(a, b, 0);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

int8x8x4_t test_vld4_lane_s8(int8_t const *a, int8x8x4_t b) {
  // CHECK-LABEL: test_vld4_lane_s8
  return vld4_lane_s8(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

int16x4x4_t test_vld4_lane_s16(int16_t const *a, int16x4x4_t b) {
  // CHECK-LABEL: test_vld4_lane_s16
  return vld4_lane_s16(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

int32x2x4_t test_vld4_lane_s32(int32_t const *a, int32x2x4_t b) {
  // CHECK-LABEL: test_vld4_lane_s32
  return vld4_lane_s32(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

int64x1x4_t test_vld4_lane_s64(int64_t const *a, int64x1x4_t b) {
  // CHECK-LABEL: test_vld4_lane_s64
  return vld4_lane_s64(a, b, 0);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

float16x4x4_t test_vld4_lane_f16(float16_t const *a, float16x4x4_t b) {
  // CHECK-LABEL: test_vld4_lane_f16
  return vld4_lane_f16(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

float32x2x4_t test_vld4_lane_f32(float32_t const *a, float32x2x4_t b) {
  // CHECK-LABEL: test_vld4_lane_f32
  return vld4_lane_f32(a, b, 1);
  // CHECK: ld4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

float64x1x4_t test_vld4_lane_f64(float64_t const *a, float64x1x4_t b) {
  // CHECK-LABEL: test_vld4_lane_f64
  return vld4_lane_f64(a, b, 0);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

poly8x8x4_t test_vld4_lane_p8(poly8_t const *a, poly8x8x4_t b) {
  // CHECK-LABEL: test_vld4_lane_p8
  return vld4_lane_p8(a, b, 7);
  // CHECK: ld4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

poly16x4x4_t test_vld4_lane_p16(poly16_t const *a, poly16x4x4_t b) {
  // CHECK-LABEL: test_vld4_lane_p16
  return vld4_lane_p16(a, b, 3);
  // CHECK: ld4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

poly64x1x4_t test_vld4_lane_p64(poly64_t const *a, poly64x1x4_t b) {
  // CHECK-LABEL: test_vld4_lane_p64
  return vld4_lane_p64(a, b, 0);
  // CHECK: ld4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_u8(uint8_t const *a, uint8x16_t b) {
  // CHECK-LABEL: test_vst1q_lane_u8
  vst1q_lane_u8(a, b, 15);
  // CHECK: st1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_u16(uint16_t const *a, uint16x8_t b) {
  // CHECK-LABEL: test_vst1q_lane_u16
  vst1q_lane_u16(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_u32(uint32_t const *a, uint32x4_t b) {
  // CHECK-LABEL: test_vst1q_lane_u32
  vst1q_lane_u32(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_u64(uint64_t const *a, uint64x2_t b) {
  // CHECK-LABEL: test_vst1q_lane_u64
  vst1q_lane_u64(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_s8(int8_t const *a, int8x16_t b) {
  // CHECK-LABEL: test_vst1q_lane_s8
  vst1q_lane_s8(a, b, 15);
  // CHECK: st1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_s16(int16_t const *a, int16x8_t b) {
  // CHECK-LABEL: test_vst1q_lane_s16
  vst1q_lane_s16(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_s32(int32_t const *a, int32x4_t b) {
  // CHECK-LABEL: test_vst1q_lane_s32
  vst1q_lane_s32(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_s64(int64_t const *a, int64x2_t b) {
  // CHECK-LABEL: test_vst1q_lane_s64
  vst1q_lane_s64(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_f16(float16_t const *a, float16x8_t b) {
  // CHECK-LABEL: test_vst1q_lane_f16
  vst1q_lane_f16(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_f32(float32_t const *a, float32x4_t b) {
  // CHECK-LABEL: test_vst1q_lane_f32
  vst1q_lane_f32(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_f64(float64_t const *a, float64x2_t b) {
  // CHECK-LABEL: test_vst1q_lane_f64
  vst1q_lane_f64(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

poly8x16_t test_vst1q_lane_p8(poly8_t const *a, poly8x16_t b) {
  // CHECK-LABEL: test_vst1q_lane_p8
  vst1q_lane_p8(a, b, 15);
  // CHECK: st1 {v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_p16(poly16_t const *a, poly16x8_t b) {
  // CHECK-LABEL: test_vst1q_lane_p16
  vst1q_lane_p16(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst1q_lane_p64(poly64_t const *a, poly64x2_t b) {
  // CHECK-LABEL: test_vst1q_lane_p64
  vst1q_lane_p64(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_u8(uint8_t const *a, uint8x8_t b) {
  // CHECK-LABEL: test_vst1_lane_u8
  vst1_lane_u8(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_u16(uint16_t const *a, uint16x4_t b) {
  // CHECK-LABEL: test_vst1_lane_u16
  vst1_lane_u16(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_u32(uint32_t const *a, uint32x2_t b) {
  // CHECK-LABEL: test_vst1_lane_u32
  vst1_lane_u32(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_u64(uint64_t const *a, uint64x1_t b) {
  // CHECK-LABEL: test_vst1_lane_u64
  vst1_lane_u64(a, b, 0);
  // CHECK: st1 {v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_s8(int8_t const *a, int8x8_t b) {
  // CHECK-LABEL: test_vst1_lane_s8
  vst1_lane_s8(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_s16(int16_t const *a, int16x4_t b) {
  // CHECK-LABEL: test_vst1_lane_s16
  vst1_lane_s16(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_s32(int32_t const *a, int32x2_t b) {
  // CHECK-LABEL: test_vst1_lane_s32
  vst1_lane_s32(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_s64(int64_t const *a, int64x1_t b) {
  // CHECK-LABEL: test_vst1_lane_s64
  vst1_lane_s64(a, b, 0);
  // CHECK: st1 {v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_f16(float16_t const *a, float16x4_t b) {
  // CHECK-LABEL: test_vst1_lane_f16
  vst1_lane_f16(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_f32(float32_t const *a, float32x2_t b) {
  // CHECK-LABEL: test_vst1_lane_f32
  vst1_lane_f32(a, b, 1);
  // CHECK: st1 {v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_f64(float64_t const *a, float64x1_t b) {
  // CHECK-LABEL: test_vst1_lane_f64
  vst1_lane_f64(a, b, 0);
  // CHECK: st1 {v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_p8(poly8_t const *a, poly8x8_t b) {
  // CHECK-LABEL: test_vst1_lane_p8
  vst1_lane_p8(a, b, 7);
  // CHECK: st1 {v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_p16(poly16_t const *a, poly16x4_t b) {
  // CHECK-LABEL: test_vst1_lane_p16
  vst1_lane_p16(a, b, 3);
  // CHECK: st1 {v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst1_lane_p64(poly64_t const *a, poly64x1_t b) {
  // CHECK-LABEL: test_vst1_lane_p64
  vst1_lane_p64(a, b, 0);
  // CHECK: st1 {v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_u8(uint8_t const *a, uint8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_u8
  vst2q_lane_u8(a, b, 15);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_u16(uint16_t const *a, uint16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_u16
  vst2q_lane_u16(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_u32(uint32_t const *a, uint32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_u32
  vst2q_lane_u32(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_u64(uint64_t const *a, uint64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_u64
  vst2q_lane_u64(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_s8(int8_t const *a, int8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_s8
  vst2q_lane_s8(a, b, 15);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_s16(int16_t const *a, int16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_s16
  vst2q_lane_s16(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_s32(int32_t const *a, int32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_s32
  vst2q_lane_s32(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_s64(int64_t const *a, int64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_s64
  vst2q_lane_s64(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_f16(float16_t const *a, float16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_f16
  vst2q_lane_f16(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_f32(float32_t const *a, float32x4x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_f32
  vst2q_lane_f32(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_f64(float64_t const *a, float64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_f64
  vst2q_lane_f64(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_p8(poly8_t const *a, poly8x16x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_p8
  vst2q_lane_p8(a, b, 15);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_p16(poly16_t const *a, poly16x8x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_p16
  vst2q_lane_p16(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst2q_lane_p64(poly64_t const *a, poly64x2x2_t b) {
  // CHECK-LABEL: test_vst2q_lane_p64
  vst2q_lane_p64(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_u8(uint8_t const *a, uint8x8x2_t b) {
  // CHECK-LABEL: test_vst2_lane_u8
  vst2_lane_u8(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_u16(uint16_t const *a, uint16x4x2_t b) {
  // CHECK-LABEL: test_vst2_lane_u16
  vst2_lane_u16(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_u32(uint32_t const *a, uint32x2x2_t b) {
  // CHECK-LABEL: test_vst2_lane_u32
  vst2_lane_u32(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_u64(uint64_t const *a, uint64x1x2_t b) {
  // CHECK-LABEL: test_vst2_lane_u64
  vst2_lane_u64(a, b, 0);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_s8(int8_t const *a, int8x8x2_t b) {
  // CHECK-LABEL: test_vst2_lane_s8
  vst2_lane_s8(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_s16(int16_t const *a, int16x4x2_t b) {
  // CHECK-LABEL: test_vst2_lane_s16
  vst2_lane_s16(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_s32(int32_t const *a, int32x2x2_t b) {
  // CHECK-LABEL: test_vst2_lane_s32
  vst2_lane_s32(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_s64(int64_t const *a, int64x1x2_t b) {
  // CHECK-LABEL: test_vst2_lane_s64
  vst2_lane_s64(a, b, 0);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_f16(float16_t const *a, float16x4x2_t b) {
  // CHECK-LABEL: test_vst2_lane_f16
  vst2_lane_f16(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_f32(float32_t const *a, float32x2x2_t b) {
  // CHECK-LABEL: test_vst2_lane_f32
  vst2_lane_f32(a, b, 1);
  // CHECK: st2 {v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_f64(float64_t const *a, float64x1x2_t b) {
  // CHECK-LABEL: test_vst2_lane_f64
  vst2_lane_f64(a, b, 0);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_p8(poly8_t const *a, poly8x8x2_t b) {
  // CHECK-LABEL: test_vst2_lane_p8
  vst2_lane_p8(a, b, 7);
  // CHECK: st2 {v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_p16(poly16_t const *a, poly16x4x2_t b) {
  // CHECK-LABEL: test_vst2_lane_p16
  vst2_lane_p16(a, b, 3);
  // CHECK: st2 {v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst2_lane_p64(poly64_t const *a, poly64x1x2_t b) {
  // CHECK-LABEL: test_vst2_lane_p64
  vst2_lane_p64(a, b, 0);
  // CHECK: st2 {v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_u8(uint8_t const *a, uint8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_u8
  vst3q_lane_u8(a, b, 15);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_u16(uint16_t const *a, uint16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_u16
  vst3q_lane_u16(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_u32(uint32_t const *a, uint32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_u32
  vst3q_lane_u32(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_u64(uint64_t const *a, uint64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_u64
  vst3q_lane_u64(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_s8(int8_t const *a, int8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_s8
  vst3q_lane_s8(a, b, 15);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_s16(int16_t const *a, int16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_s16
  vst3q_lane_s16(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_s32(int32_t const *a, int32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_s32
  vst3q_lane_s32(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_s64(int64_t const *a, int64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_s64
  vst3q_lane_s64(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_f16(float16_t const *a, float16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_f16
  vst3q_lane_f16(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_f32(float32_t const *a, float32x4x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_f32
  vst3q_lane_f32(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_f64(float64_t const *a, float64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_f64
  vst3q_lane_f64(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_p8(poly8_t const *a, poly8x16x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_p8
  vst3q_lane_p8(a, b, 15);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_p16(poly16_t const *a, poly16x8x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_p16
  vst3q_lane_p16(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst3q_lane_p64(poly64_t const *a, poly64x2x3_t b) {
  // CHECK-LABEL: test_vst3q_lane_p64
  vst3q_lane_p64(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_u8(uint8_t const *a, uint8x8x3_t b) {
  // CHECK-LABEL: test_vst3_lane_u8
  vst3_lane_u8(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_u16(uint16_t const *a, uint16x4x3_t b) {
  // CHECK-LABEL: test_vst3_lane_u16
  vst3_lane_u16(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_u32(uint32_t const *a, uint32x2x3_t b) {
  // CHECK-LABEL: test_vst3_lane_u32
  vst3_lane_u32(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_u64(uint64_t const *a, uint64x1x3_t b) {
  // CHECK-LABEL: test_vst3_lane_u64
  vst3_lane_u64(a, b, 0);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_s8(int8_t const *a, int8x8x3_t b) {
  // CHECK-LABEL: test_vst3_lane_s8
  vst3_lane_s8(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_s16(int16_t const *a, int16x4x3_t b) {
  // CHECK-LABEL: test_vst3_lane_s16
  vst3_lane_s16(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_s32(int32_t const *a, int32x2x3_t b) {
  // CHECK-LABEL: test_vst3_lane_s32
  vst3_lane_s32(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_s64(int64_t const *a, int64x1x3_t b) {
  // CHECK-LABEL: test_vst3_lane_s64
  vst3_lane_s64(a, b, 0);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_f16(float16_t const *a, float16x4x3_t b) {
  // CHECK-LABEL: test_vst3_lane_f16
  vst3_lane_f16(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_f32(float32_t const *a, float32x2x3_t b) {
  // CHECK-LABEL: test_vst3_lane_f32
  vst3_lane_f32(a, b, 1);
  // CHECK: st3 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_f64(float64_t const *a, float64x1x3_t b) {
  // CHECK-LABEL: test_vst3_lane_f64
  vst3_lane_f64(a, b, 0);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_p8(poly8_t const *a, poly8x8x3_t b) {
  // CHECK-LABEL: test_vst3_lane_p8
  vst3_lane_p8(a, b, 7);
  // CHECK: st3 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_p16(poly16_t const *a, poly16x4x3_t b) {
  // CHECK-LABEL: test_vst3_lane_p16
  vst3_lane_p16(a, b, 3);
  // CHECK: st3 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst3_lane_p64(poly64_t const *a, poly64x1x3_t b) {
  // CHECK-LABEL: test_vst3_lane_p64
  vst3_lane_p64(a, b, 0);
  // CHECK: st3 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_u8(uint16_t const *a, uint8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_u8
  vst4q_lane_u8(a, b, 15);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_u16(uint16_t const *a, uint16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_u16
  vst4q_lane_u16(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_u32(uint32_t const *a, uint32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_u32
  vst4q_lane_u32(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_u64(uint64_t const *a, uint64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_u64
  vst4q_lane_u64(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_s8(int16_t const *a, int8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_s8
  vst4q_lane_s8(a, b, 15);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_s16(int16_t const *a, int16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_s16
  vst4q_lane_s16(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_s32(int32_t const *a, int32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_s32
  vst4q_lane_s32(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_s64(int64_t const *a, int64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_s64
  vst4q_lane_s64(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_f16(float16_t const *a, float16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_f16
  vst4q_lane_f16(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_f32(float32_t const *a, float32x4x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_f32
  vst4q_lane_f32(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[3], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_f64(float64_t const *a, float64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_f64
  vst4q_lane_f64(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_p8(poly16_t const *a, poly8x16x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_p8
  vst4q_lane_p8(a, b, 15);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[15], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_p16(poly16_t const *a, poly16x8x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_p16
  vst4q_lane_p16(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[7], [{{x[0-9]+|sp}}]
}

void test_vst4q_lane_p64(poly64_t const *a, poly64x2x4_t b) {
  // CHECK-LABEL: test_vst4q_lane_p64
  vst4q_lane_p64(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[1], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_u8(uint8_t const *a, uint8x8x4_t b) {
  // CHECK-LABEL: test_vst4_lane_u8
  vst4_lane_u8(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_u16(uint16_t const *a, uint16x4x4_t b) {
  // CHECK-LABEL: test_vst4_lane_u16
  vst4_lane_u16(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_u32(uint32_t const *a, uint32x2x4_t b) {
  // CHECK-LABEL: test_vst4_lane_u32
  vst4_lane_u32(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_u64(uint64_t const *a, uint64x1x4_t b) {
  // CHECK-LABEL: test_vst4_lane_u64
  vst4_lane_u64(a, b, 0);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_s8(int8_t const *a, int8x8x4_t b) {
  // CHECK-LABEL: test_vst4_lane_s8
  vst4_lane_s8(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_s16(int16_t const *a, int16x4x4_t b) {
  // CHECK-LABEL: test_vst4_lane_s16
  vst4_lane_s16(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_s32(int32_t const *a, int32x2x4_t b) {
  // CHECK-LABEL: test_vst4_lane_s32
  vst4_lane_s32(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_s64(int64_t const *a, int64x1x4_t b) {
  // CHECK-LABEL: test_vst4_lane_s64
  vst4_lane_s64(a, b, 0);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_f16(float16_t const *a, float16x4x4_t b) {
  // CHECK-LABEL: test_vst4_lane_f16
  vst4_lane_f16(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_f32(float32_t const *a, float32x2x4_t b) {
  // CHECK-LABEL: test_vst4_lane_f32
  vst4_lane_f32(a, b, 1);
  // CHECK: st4 {v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s, v{{[0-9]+}}.s}[1], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_f64(float64_t const *a, float64x1x4_t b) {
  // CHECK-LABEL: test_vst4_lane_f64
  vst4_lane_f64(a, b, 0);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_p8(poly8_t const *a, poly8x8x4_t b) {
  // CHECK-LABEL: test_vst4_lane_p8
  vst4_lane_p8(a, b, 7);
  // CHECK: st4 {v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b, v{{[0-9]+}}.b}[7], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_p16(poly16_t const *a, poly16x4x4_t b) {
  // CHECK-LABEL: test_vst4_lane_p16
  vst4_lane_p16(a, b, 3);
  // CHECK: st4 {v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h, v{{[0-9]+}}.h}[3], [{{x[0-9]+|sp}}]
}

void test_vst4_lane_p64(poly64_t const *a, poly64x1x4_t b) {
  // CHECK-LABEL: test_vst4_lane_p64
  vst4_lane_p64(a, b, 0);
  // CHECK: st4 {v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d, v{{[0-9]+}}.d}[0], [{{x[0-9]+|sp}}]
}

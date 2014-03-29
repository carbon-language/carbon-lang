// REQUIRES: aarch64-registered-target
// REQUIRES: arm64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types
#include <arm_neon.h>

int8x8_t test_vuzp1_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vuzp1_s8
  return vuzp1_s8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vuzp1q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vuzp1q_s8
  return vuzp1q_s8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vuzp1_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vuzp1_s16
  return vuzp1_s16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vuzp1q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vuzp1q_s16
  return vuzp1q_s16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vuzp1_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vuzp1_s32
  return vuzp1_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vuzp1q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vuzp1q_s32
  return vuzp1q_s32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vuzp1q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vuzp1q_s64
  return vuzp1q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vuzp1_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vuzp1_u8
  return vuzp1_u8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vuzp1q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vuzp1q_u8
  return vuzp1q_u8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vuzp1_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vuzp1_u16
  return vuzp1_u16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vuzp1q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vuzp1q_u16
  return vuzp1q_u16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vuzp1_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vuzp1_u32
  return vuzp1_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vuzp1q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vuzp1q_u32
  return vuzp1q_u32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vuzp1q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vuzp1q_u64
  return vuzp1q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vuzp1_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vuzp1_f32
  return vuzp1_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vuzp1q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vuzp1q_f32
  return vuzp1q_f32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vuzp1q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vuzp1q_f64
  return vuzp1q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vuzp1_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vuzp1_p8
  return vuzp1_p8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vuzp1q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vuzp1q_p8
  return vuzp1q_p8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vuzp1_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vuzp1_p16
  return vuzp1_p16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vuzp1q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vuzp1q_p16
  return vuzp1q_p16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8_t test_vuzp2_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vuzp2_s8
  return vuzp2_s8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vuzp2q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vuzp2q_s8
  return vuzp2q_s8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vuzp2_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vuzp2_s16
  return vuzp2_s16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vuzp2q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vuzp2q_s16
  return vuzp2q_s16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vuzp2_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vuzp2_s32
  return vuzp2_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vuzp2q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vuzp2q_s32
  return vuzp2q_s32(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vuzp2q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vuzp2q_s64
  return vuzp2q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vuzp2_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vuzp2_u8
  return vuzp2_u8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vuzp2q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vuzp2q_u8
  return vuzp2q_u8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vuzp2_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vuzp2_u16
  return vuzp2_u16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vuzp2q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vuzp2q_u16
  return vuzp2q_u16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vuzp2_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vuzp2_u32
  return vuzp2_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vuzp2q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vuzp2q_u32
  return vuzp2q_u32(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vuzp2q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vuzp2q_u64
  return vuzp2q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vuzp2_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vuzp2_f32
  return vuzp2_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vuzp2q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vuzp2q_f32
  return vuzp2q_f32(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vuzp2q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vuzp2q_f64
  return vuzp2q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vuzp2_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vuzp2_p8
  return vuzp2_p8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vuzp2q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vuzp2q_p8
  return vuzp2q_p8(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vuzp2_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vuzp2_p16
  return vuzp2_p16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vuzp2q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vuzp2q_p16
  return vuzp2q_p16(a, b);
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8_t test_vzip1_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vzip1_s8
  return vzip1_s8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vzip1q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vzip1q_s8
  return vzip1q_s8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vzip1_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vzip1_s16
  return vzip1_s16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vzip1q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vzip1q_s16
  return vzip1q_s16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vzip1_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vzip1_s32
  return vzip1_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vzip1q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vzip1q_s32
  return vzip1q_s32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vzip1q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vzip1q_s64
  return vzip1q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vzip1_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vzip1_u8
  return vzip1_u8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vzip1q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vzip1q_u8
  return vzip1q_u8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vzip1_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vzip1_u16
  return vzip1_u16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vzip1q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vzip1q_u16
  return vzip1q_u16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vzip1_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vzip1_u32
  return vzip1_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vzip1q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vzip1q_u32
  return vzip1q_u32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vzip1q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vzip1q_u64
  return vzip1q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vzip1_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vzip1_f32
  return vzip1_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vzip1q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vzip1q_f32
  return vzip1q_f32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vzip1q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vzip1q_f64
  return vzip1q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vzip1_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vzip1_p8
  return vzip1_p8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vzip1q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vzip1q_p8
  return vzip1q_p8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vzip1_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vzip1_p16
  return vzip1_p16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vzip1q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vzip1q_p16
  return vzip1q_p16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8_t test_vzip2_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vzip2_s8
  return vzip2_s8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vzip2q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vzip2q_s8
  return vzip2q_s8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vzip2_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vzip2_s16
  return vzip2_s16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vzip2q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vzip2q_s16
  return vzip2q_s16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vzip2_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vzip2_s32
  return vzip2_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vzip2q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vzip2q_s32
  return vzip2q_s32(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vzip2q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vzip2q_s64
  return vzip2q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vzip2_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vzip2_u8
  return vzip2_u8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vzip2q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vzip2q_u8
  return vzip2q_u8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vzip2_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vzip2_u16
  return vzip2_u16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vzip2q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vzip2q_u16
  return vzip2q_u16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vzip2_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vzip2_u32
  return vzip2_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vzip2q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vzip2q_u32
  return vzip2q_u32(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vzip2q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vzip2q_u64
  return vzip2q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vzip2_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vzip2_f32
  return vzip2_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vzip2q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vzip2q_f32
  return vzip2q_f32(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vzip2q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vzip2q_f64
  return vzip2q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vzip2_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vzip2_p8
  return vzip2_p8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vzip2q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vzip2q_p8
  return vzip2q_p8(a, b);
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vzip2_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vzip2_p16
  return vzip2_p16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vzip2q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vzip2q_p16
  return vzip2q_p16(a, b);
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8_t test_vtrn1_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtrn1_s8
  return vtrn1_s8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vtrn1q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vtrn1q_s8
  return vtrn1q_s8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vtrn1_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vtrn1_s16
  return vtrn1_s16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vtrn1q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vtrn1q_s16
  return vtrn1q_s16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vtrn1_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vtrn1_s32
  return vtrn1_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vtrn1q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vtrn1q_s32
  return vtrn1q_s32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vtrn1q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vtrn1q_s64
  return vtrn1q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vtrn1_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtrn1_u8
  return vtrn1_u8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vtrn1q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vtrn1q_u8
  return vtrn1q_u8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vtrn1_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vtrn1_u16
  return vtrn1_u16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vtrn1q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vtrn1q_u16
  return vtrn1q_u16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vtrn1_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vtrn1_u32
  return vtrn1_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vtrn1q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vtrn1q_u32
  return vtrn1q_u32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vtrn1q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vtrn1q_u64
  return vtrn1q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vtrn1_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vtrn1_f32
  return vtrn1_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vtrn1q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vtrn1q_f32
  return vtrn1q_f32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vtrn1q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vtrn1q_f64
  return vtrn1q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[1\], v[0-9]+.d\[0\]|zip1 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vtrn1_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vtrn1_p8
  return vtrn1_p8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vtrn1q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vtrn1q_p8
  return vtrn1q_p8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vtrn1_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vtrn1_p16
  return vtrn1_p16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vtrn1q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vtrn1q_p16
  return vtrn1q_p16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8_t test_vtrn2_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtrn2_s8
  return vtrn2_s8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vtrn2q_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vtrn2q_s8
  return vtrn2q_s8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vtrn2_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vtrn2_s16
  return vtrn2_s16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int16x8_t test_vtrn2q_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vtrn2q_s16
  return vtrn2q_s16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x2_t test_vtrn2_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vtrn2_s32
  return vtrn2_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

int32x4_t test_vtrn2q_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vtrn2q_s32
  return vtrn2q_s32(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vtrn2q_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vtrn2q_s64
  return vtrn2q_s64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

uint8x8_t test_vtrn2_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtrn2_u8
  return vtrn2_u8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vtrn2q_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vtrn2q_u8
  return vtrn2q_u8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vtrn2_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vtrn2_u16
  return vtrn2_u16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint16x8_t test_vtrn2q_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vtrn2q_u16
  return vtrn2q_u16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x2_t test_vtrn2_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vtrn2_u32
  return vtrn2_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

uint32x4_t test_vtrn2q_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vtrn2q_u32
  return vtrn2q_u32(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vtrn2q_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vtrn2q_u64
  return vtrn2q_u64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

float32x2_t test_vtrn2_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vtrn2_f32
  return vtrn2_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v0.2s, v0.2s, v1.2s}}
}

float32x4_t test_vtrn2q_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vtrn2q_f32
  return vtrn2q_f32(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vtrn2q_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vtrn2q_f64
  return vtrn2q_f64(a, b);
  // CHECK: {{ins v[0-9]+.d\[0\], v[0-9]+.d\[1\]|zip2 v0.2d, v0.2d, v1.2d}}
}

poly8x8_t test_vtrn2_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vtrn2_p8
  return vtrn2_p8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vtrn2q_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vtrn2q_p8
  return vtrn2q_p8(a, b);
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x4_t test_vtrn2_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vtrn2_p16
  return vtrn2_p16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

poly16x8_t test_vtrn2q_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vtrn2q_p16
  return vtrn2q_p16(a, b);
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8x2_t test_vuzp_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vuzp_s8
  return vuzp_s8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4x2_t test_vuzp_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vuzp_s16
  return vuzp_s16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int32x2x2_t test_vuzp_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vuzp_s32
  return vuzp_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
uint8x8x2_t test_vuzp_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vuzp_u8
  return vuzp_u8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint16x4x2_t test_vuzp_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vuzp_u16
  return vuzp_u16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint32x2x2_t test_vuzp_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vuzp_u32
  return vuzp_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
float32x2x2_t test_vuzp_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vuzp_f32
  return vuzp_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
poly8x8x2_t test_vuzp_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vuzp_p8
  return vuzp_p8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: uzp2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
poly16x4x2_t test_vuzp_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vuzp_p16
  return vuzp_p16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: uzp2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int8x16x2_t test_vuzpq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vuzpq_s8
  return vuzpq_s8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int16x8x2_t test_vuzpq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vuzpq_s16
  return vuzpq_s16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int32x4x2_t test_vuzpq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vuzpq_s32
  return vuzpq_s32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint8x16x2_t test_vuzpq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vuzpq_u8
  return vuzpq_u8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint16x8x2_t test_vuzpq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vuzpq_u16
  return vuzpq_u16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint32x4x2_t test_vuzpq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vuzpq_u32
  return vuzpq_u32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
float32x4x2_t test_vuzpq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vuzpq_f32
  return vuzpq_f32(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: uzp2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
poly8x16x2_t test_vuzpq_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vuzpq_p8
  return vuzpq_p8(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: uzp2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
poly16x8x2_t test_vuzpq_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vuzpq_p16
  return vuzpq_p16(a, b);
  // CHECK: uzp1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: uzp2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8x2_t test_vzip_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vzip_s8
  return vzip_s8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4x2_t test_vzip_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vzip_s16
  return vzip_s16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int32x2x2_t test_vzip_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vzip_s32
  return vzip_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
uint8x8x2_t test_vzip_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vzip_u8
  return vzip_u8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint16x4x2_t test_vzip_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vzip_u16
  return vzip_u16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint32x2x2_t test_vzip_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vzip_u32
  return vzip_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
float32x2x2_t test_vzip_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vzip_f32
  return vzip_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
poly8x8x2_t test_vzip_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vzip_p8
  return vzip_p8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: zip2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
poly16x4x2_t test_vzip_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vzip_p16
  return vzip_p16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: zip2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int8x16x2_t test_vzipq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vzipq_s8
  return vzipq_s8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int16x8x2_t test_vzipq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vzipq_s16
  return vzipq_s16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int32x4x2_t test_vzipq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vzipq_s32
  return vzipq_s32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint8x16x2_t test_vzipq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vzipq_u8
  return vzipq_u8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint16x8x2_t test_vzipq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vzipq_u16
  return vzipq_u16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint32x4x2_t test_vzipq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vzipq_u32
  return vzipq_u32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
float32x4x2_t test_vzipq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vzipq_f32
  return vzipq_f32(a, b);
  // CHECK: zip1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: zip2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
poly8x16x2_t test_vzipq_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vzipq_p8
  return vzipq_p8(a, b);
  // CHECK: zip1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: zip2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
poly16x8x2_t test_vzipq_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vzipq_p16
  return vzipq_p16(a, b);
  // CHECK: zip1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: zip2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int8x8x2_t test_vtrn_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vtrn_s8
  return vtrn_s8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4x2_t test_vtrn_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vtrn_s16
  return vtrn_s16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int32x2x2_t test_vtrn_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vtrn_s32
  return vtrn_s32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
uint8x8x2_t test_vtrn_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vtrn_u8
  return vtrn_u8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
uint16x4x2_t test_vtrn_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vtrn_u16
  return vtrn_u16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
uint32x2x2_t test_vtrn_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vtrn_u32
  return vtrn_u32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
float32x2x2_t test_vtrn_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vtrn_f32
  return vtrn_f32(a, b);
  // CHECK: {{ins v[0-9]+.s\[1\], v[0-9]+.s\[0\]|zip1 v2.2s, v0.2s, v1.2s}}
  // CHECK: {{ins v[0-9]+.s\[0\], v[0-9]+.s\[1\]|zip2 v1.2s, v0.2s, v1.2s}}
}
poly8x8x2_t test_vtrn_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vtrn_p8
  return vtrn_p8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
  // CHECK: trn2 {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
poly16x4x2_t test_vtrn_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vtrn_p16
  return vtrn_p16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
  // CHECK: trn2 {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int8x16x2_t test_vtrnq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vtrnq_s8
  return vtrnq_s8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
int16x8x2_t test_vtrnq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vtrnq_s16
  return vtrnq_s16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
int32x4x2_t test_vtrnq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vtrnq_s32
  return vtrnq_s32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint8x16x2_t test_vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vtrnq_u8
  return vtrnq_u8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
uint16x8x2_t test_vtrnq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vtrnq_u16
  return vtrnq_u16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}
uint32x4x2_t test_vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vtrnq_u32
  return vtrnq_u32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
float32x4x2_t test_vtrnq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vtrnq_f32
  return vtrnq_f32(a, b);
  // CHECK: trn1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK: trn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
poly8x16x2_t test_vtrnq_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vtrnq_p8
  return vtrnq_p8(a, b);
  // CHECK: trn1 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
  // CHECK: trn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}
poly16x8x2_t test_vtrnq_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vtrnq_p16
  return vtrnq_p16(a, b);
  // CHECK: trn1 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
  // CHECK: trn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

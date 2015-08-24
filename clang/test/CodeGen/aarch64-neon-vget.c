// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-apple-darwin -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

#include <arm_neon.h>

uint8_t test_vget_lane_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vget_lane_u8:
  // CHECK-NEXT:  umov.b w0, v0[7]
  // CHECK-NEXT:  ret
  return vget_lane_u8(a, 7);
}

uint16_t test_vget_lane_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vget_lane_u16:
  // CHECK-NEXT:  umov.h w0, v0[3]
  // CHECK-NEXT:  ret
  return vget_lane_u16(a, 3);
}

uint32_t test_vget_lane_u32(uint32x2_t a) {
  // CHECK-LABEL: test_vget_lane_u32:
  // CHECK-NEXT:  mov.s  w0, v0[1]
  // CHECK-NEXT:  ret
  return vget_lane_u32(a, 1);
}

int8_t test_vget_lane_s8(int8x8_t a) {
  // CHECK-LABEL: test_vget_lane_s8:
  // CHECK-NEXT:  umov.b w0, v0[7]
  // CHECK-NEXT:  ret
  return vget_lane_s8(a, 7);
}

int16_t test_vget_lane_s16(int16x4_t a) {
  // CHECK-LABEL: test_vget_lane_s16:
  // CHECK-NEXT:  umov.h w0, v0[3]
  // CHECK-NEXT:  ret
  return vget_lane_s16(a, 3);
}

int32_t test_vget_lane_s32(int32x2_t a) {
  // CHECK-LABEL: test_vget_lane_s32:
  // CHECK-NEXT:  mov.s  w0, v0[1]
  // CHECK-NEXT:  ret
  return vget_lane_s32(a, 1);
}

poly8_t test_vget_lane_p8(poly8x8_t a) {
  // CHECK-LABEL: test_vget_lane_p8:
  // CHECK-NEXT:  umov.b w0, v0[7]
  // CHECK-NEXT:  ret
  return vget_lane_p8(a, 7);
}

poly16_t test_vget_lane_p16(poly16x4_t a) {
  // CHECK-LABEL: test_vget_lane_p16:
  // CHECK-NEXT:  umov.h w0, v0[3]
  // CHECK-NEXT:  ret
  return vget_lane_p16(a, 3);
}

float32_t test_vget_lane_f32(float32x2_t a) {
  // CHECK-LABEL: test_vget_lane_f32:
  // CHECK-NEXT:  mov s0, v0[1]
  // CHECK-NEXT:  ret
  return vget_lane_f32(a, 1);
}

float32_t test_vget_lane_f16(float16x4_t a) {
  // CHECK-LABEL: test_vget_lane_f16:
  // CHECK-NEXT:  umov.h w8, v0[1]
  // CHECK-NEXT:  fmov s0, w8
  // CHECK-NEXT:  fcvt s0, h0
  // CHECK-NEXT:  ret
  return vget_lane_f16(a, 1);
}

uint8_t test_vgetq_lane_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vgetq_lane_u8:
  // CHECK-NEXT:  umov.b w0, v0[15]
  // CHECK-NEXT:  ret
  return vgetq_lane_u8(a, 15);
}

uint16_t test_vgetq_lane_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vgetq_lane_u16:
  // CHECK-NEXT:  umov.h w0, v0[7]
  // CHECK-NEXT:  ret
  return vgetq_lane_u16(a, 7);
}

uint32_t test_vgetq_lane_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vgetq_lane_u32:
  // CHECK-NEXT:  mov.s  w0, v0[3]
  // CHECK-NEXT:  ret
  return vgetq_lane_u32(a, 3);
}

int8_t test_vgetq_lane_s8(int8x16_t a) {
  // CHECK-LABEL: test_vgetq_lane_s8:
  // CHECK-NEXT:  umov.b w0, v0[15]
  // CHECK-NEXT:  ret
  return vgetq_lane_s8(a, 15);
}

int16_t test_vgetq_lane_s16(int16x8_t a) {
  // CHECK-LABEL: test_vgetq_lane_s16:
  // CHECK-NEXT:  umov.h w0, v0[7]
  // CHECK-NEXT:  ret
  return vgetq_lane_s16(a, 7);
}

int32_t test_vgetq_lane_s32(int32x4_t a) {
  // CHECK-LABEL: test_vgetq_lane_s32:
  // CHECK-NEXT:  mov.s  w0, v0[3]
  // CHECK-NEXT:  ret
  return vgetq_lane_s32(a, 3);
}

poly8_t test_vgetq_lane_p8(poly8x16_t a) {
  // CHECK-LABEL: test_vgetq_lane_p8:
  // CHECK-NEXT:  umov.b w0, v0[15]
  // CHECK-NEXT:  ret
  return vgetq_lane_p8(a, 15);
}

poly16_t test_vgetq_lane_p16(poly16x8_t a) {
  // CHECK-LABEL: test_vgetq_lane_p16:
  // CHECK-NEXT:  umov.h w0, v0[7]
  // CHECK-NEXT:  ret
  return vgetq_lane_p16(a, 7);
}

float32_t test_vgetq_lane_f32(float32x4_t a) {
  // CHECK-LABEL: test_vgetq_lane_f32:
  // CHECK-NEXT:  mov s0, v0[3]
  // CHECK-NEXT:  ret
  return vgetq_lane_f32(a, 3);
}

float32_t test_vgetq_lane_f16(float16x8_t a) {
  // CHECK-LABEL: test_vgetq_lane_f16:
  // CHECK-NEXT:  umov.h w8, v0[3]
  // CHECK-NEXT:  fmov s0, w8
  // CHECK-NEXT:  fcvt s0, h0
  // CHECK-NEXT:  ret
  return vgetq_lane_f16(a, 3);
}

int64_t test_vget_lane_s64(int64x1_t a) {
  // CHECK-LABEL: test_vget_lane_s64:
  // CHECK-NEXT:  fmov x0, d0
  // CHECK-NEXT:  ret
  return vget_lane_s64(a, 0);
}

uint64_t test_vget_lane_u64(uint64x1_t a) {
  // CHECK-LABEL: test_vget_lane_u64:
  // CHECK-NEXT:  fmov x0, d0
  // CHECK-NEXT:  ret
  return vget_lane_u64(a, 0);
}

int64_t test_vgetq_lane_s64(int64x2_t a) {
  // CHECK-LABEL: test_vgetq_lane_s64:
  // CHECK-NEXT:  mov.d  x0, v0[1]
  // CHECK-NEXT:  ret
  return vgetq_lane_s64(a, 1);
}

uint64_t test_vgetq_lane_u64(uint64x2_t a) {
  // CHECK-LABEL: test_vgetq_lane_u64:
  // CHECK-NEXT:  mov.d  x0, v0[1]
  // CHECK-NEXT:  ret
  return vgetq_lane_u64(a, 1);
}


uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vset_lane_u8:
  // CHECK-NEXT:  ins.b v0[7], w0
  // CHECK-NEXT:  ret
  return vset_lane_u8(a, b, 7);
}

uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vset_lane_u16:
  // CHECK-NEXT:  ins.h v0[3], w0
  // CHECK-NEXT:  ret
  return vset_lane_u16(a, b, 3);
}

uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vset_lane_u32:
  // CHECK-NEXT:  ins.s v0[1], w0
  // CHECK-NEXT:  ret
  return vset_lane_u32(a, b, 1);
}

int8x8_t test_vset_lane_s8(int8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vset_lane_s8:
  // CHECK-NEXT:  ins.b v0[7], w0
  // CHECK-NEXT:  ret
  return vset_lane_s8(a, b, 7);
}

int16x4_t test_vset_lane_s16(int16_t a, int16x4_t b) {
  // CHECK-LABEL: test_vset_lane_s16:
  // CHECK-NEXT:  ins.h v0[3], w0
  // CHECK-NEXT:  ret
  return vset_lane_s16(a, b, 3);
}

int32x2_t test_vset_lane_s32(int32_t a, int32x2_t b) {
  // CHECK-LABEL: test_vset_lane_s32:
  // CHECK-NEXT:  ins.s v0[1], w0
  // CHECK-NEXT:  ret
  return vset_lane_s32(a, b, 1);
}

poly8x8_t test_vset_lane_p8(poly8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vset_lane_p8:
  // CHECK-NEXT:  ins.b v0[7], w0
  // CHECK-NEXT:  ret
  return vset_lane_p8(a, b, 7);
}

poly16x4_t test_vset_lane_p16(poly16_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vset_lane_p16:
  // CHECK-NEXT:  ins.h v0[3], w0
  // CHECK-NEXT:  ret
  return vset_lane_p16(a, b, 3);
}

float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
  // CHECK-LABEL: test_vset_lane_f32:
  // CHECK-NEXT:  ins.s v1[1], v0[0]
  // CHECK-NEXT:  mov.16b  v0, v1
  // CHECK-NEXT:  ret
  return vset_lane_f32(a, b, 1);
}

float16x4_t test_vset_lane_f16(float16_t *a, float16x4_t b) {
  // CHECK-LABEL: test_vset_lane_f16:
  // CHECK-NEXT:  ld1.h { v0 }[3], [x0]
  // CHECK-NEXT:  ret
  return vset_lane_f16(*a, b, 3);
}

uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vsetq_lane_u8:
  // CHECK-NEXT:  ins.b v0[15], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_u8(a, b, 15);
}

uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vsetq_lane_u16:
  // CHECK-NEXT:  ins.h v0[7], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_u16(a, b, 7);
}

uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vsetq_lane_u32:
  // CHECK-NEXT:  ins.s v0[3], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_u32(a, b, 3);
}

int8x16_t test_vsetq_lane_s8(int8_t a, int8x16_t b) {
  // CHECK-LABEL: test_vsetq_lane_s8:
  // CHECK-NEXT:  ins.b v0[15], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_s8(a, b, 15);
}

int16x8_t test_vsetq_lane_s16(int16_t a, int16x8_t b) {
  // CHECK-LABEL: test_vsetq_lane_s16:
  // CHECK-NEXT:  ins.h v0[7], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_s16(a, b, 7);
}

int32x4_t test_vsetq_lane_s32(int32_t a, int32x4_t b) {
  // CHECK-LABEL: test_vsetq_lane_s32:
  // CHECK-NEXT:  ins.s v0[3], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_s32(a, b, 3);
}

poly8x16_t test_vsetq_lane_p8(poly8_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vsetq_lane_p8:
  // CHECK-NEXT:  ins.b v0[15], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_p8(a, b, 15);
}

poly16x8_t test_vsetq_lane_p16(poly16_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vsetq_lane_p16:
  // CHECK-NEXT:  ins.h v0[7], w0
  // CHECK-NEXT:  ret
  return vsetq_lane_p16(a, b, 7);
}

float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
  // CHECK-LABEL: test_vsetq_lane_f32:
  // CHECK-NEXT:  ins.s v1[3], v0[0]
  // CHECK-NEXT:  mov.16b  v0, v1
  // CHECK-NEXT:  ret
  return vsetq_lane_f32(a, b, 3);
}

float16x8_t test_vsetq_lane_f16(float16_t *a, float16x8_t b) {
  // CHECK-LABEL: test_vsetq_lane_f16:
  // CHECK-NEXT:  ld1.h { v0 }[7], [x0]
  // CHECK-NEXT:  ret
  return vsetq_lane_f16(*a, b, 7);
}

int64x1_t test_vset_lane_s64(int64_t a, int64x1_t b) {
  // CHECK-LABEL: test_vset_lane_s64:
  // CHECK-NEXT:  fmov d0, x0
  // CHECK-NEXT:  ret
  return vset_lane_s64(a, b, 0);
}

uint64x1_t test_vset_lane_u64(uint64_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vset_lane_u64:
  // CHECK-NEXT:  fmov d0, x0
  // CHECK-NEXT:  ret
  return vset_lane_u64(a, b, 0);
}

int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
  // CHECK-LABEL: test_vsetq_lane_s64:
  // CHECK-NEXT:  ins.d v0[1], x0
  // CHECK-NEXT:  ret
  return vsetq_lane_s64(a, b, 1);
}

uint64x2_t test_vsetq_lane_u64(uint64_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vsetq_lane_u64:
  // CHECK-NEXT:  ins.d v0[1], x0
  // CHECK-NEXT:  ret
  return vsetq_lane_u64(a, b, 1);
}

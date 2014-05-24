// REQUIRES: arm64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vext_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vext_s8
  return vext_s8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?2}}
}

int16x4_t test_vext_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vext_s16
  return vext_s16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?6}}
}

int32x2_t test_vext_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vext_s32
  return vext_s32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?4}}
}

int64x1_t test_vext_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vext_s64
  return vext_s64(a, b, 0);
}

int8x16_t test_vextq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vextq_s8
  return vextq_s8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?2}}
}

int16x8_t test_vextq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vextq_s16
  return vextq_s16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?6}}
}

int32x4_t test_vextq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vextq_s32
  return vextq_s32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?4}}
}

int64x2_t test_vextq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vextq_s64
  return vextq_s64(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?8}}
}

uint8x8_t test_vext_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vext_u8
  return vext_u8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?2}}
}

uint16x4_t test_vext_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vext_u16
  return vext_u16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?6}}
}

uint32x2_t test_vext_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vext_u32
  return vext_u32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?4}}
}

uint64x1_t test_vext_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vext_u64
  return vext_u64(a, b, 0);
}

uint8x16_t test_vextq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vextq_u8
  return vextq_u8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?2}}
}

uint16x8_t test_vextq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vextq_u16
  return vextq_u16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?6}}
}

uint32x4_t test_vextq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vextq_u32
  return vextq_u32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?4}}
}

uint64x2_t test_vextq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vextq_u64
  return vextq_u64(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?8}}
}

float32x2_t test_vext_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vext_f32
  return vext_f32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?4}}
}

float64x1_t test_vext_f64(float64x1_t a, float64x1_t b) {
  // CHECK-LABEL: test_vext_f64
  return vext_f64(a, b, 0);
}

float32x4_t test_vextq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vextq_f32
  return vextq_f32(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?4}}
}

float64x2_t test_vextq_f64(float64x2_t a, float64x2_t b) {
  // CHECK-LABEL: test_vextq_f64
  return vextq_f64(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?8}}
}

poly8x8_t test_vext_p8(poly8x8_t a, poly8x8_t b) {
  // CHECK-LABEL: test_vext_p8
  return vext_p8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?2}}
}

poly16x4_t test_vext_p16(poly16x4_t a, poly16x4_t b) {
  // CHECK-LABEL: test_vext_p16
  return vext_p16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{(0x)?6}}
}

poly8x16_t test_vextq_p8(poly8x16_t a, poly8x16_t b) {
  // CHECK-LABEL: test_vextq_p8
  return vextq_p8(a, b, 2);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?2}}
}

poly16x8_t test_vextq_p16(poly16x8_t a, poly16x8_t b) {
  // CHECK-LABEL: test_vextq_p16
  return vextq_p16(a, b, 3);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{(0x)?6}}
}

// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int16_t test_vaddlv_s8(int8x8_t a) {
  // CHECK-LABEL: test_vaddlv_s8
  return vaddlv_s8(a);
  // CHECK: saddlv {{h[0-9]+}}, {{v[0-9]+}}.8b
}

int32_t test_vaddlv_s16(int16x4_t a) {
  // CHECK-LABEL: test_vaddlv_s16
  return vaddlv_s16(a);
  // CHECK: saddlv {{s[0-9]+}}, {{v[0-9]+}}.4h
}

uint16_t test_vaddlv_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vaddlv_u8
  return vaddlv_u8(a);
  // CHECK: uaddlv {{h[0-9]+}}, {{v[0-9]+}}.8b
}

uint32_t test_vaddlv_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vaddlv_u16
  return vaddlv_u16(a);
  // CHECK: uaddlv {{s[0-9]+}}, {{v[0-9]+}}.4h
}

int16_t test_vaddlvq_s8(int8x16_t a) {
  // CHECK-LABEL: test_vaddlvq_s8
  return vaddlvq_s8(a);
  // CHECK: saddlv {{h[0-9]+}}, {{v[0-9]+}}.16b
}

int32_t test_vaddlvq_s16(int16x8_t a) {
  // CHECK-LABEL: test_vaddlvq_s16
  return vaddlvq_s16(a);
  // CHECK: saddlv {{s[0-9]+}}, {{v[0-9]+}}.8h
}

int64_t test_vaddlvq_s32(int32x4_t a) {
  // CHECK-LABEL: test_vaddlvq_s32
  return vaddlvq_s32(a);
  // CHECK: saddlv {{d[0-9]+}}, {{v[0-9]+}}.4s
}

uint16_t test_vaddlvq_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vaddlvq_u8
  return vaddlvq_u8(a);
  // CHECK: uaddlv {{h[0-9]+}}, {{v[0-9]+}}.16b
}

uint32_t test_vaddlvq_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vaddlvq_u16
  return vaddlvq_u16(a);
  // CHECK: uaddlv {{s[0-9]+}}, {{v[0-9]+}}.8h
}

uint64_t test_vaddlvq_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vaddlvq_u32
  return vaddlvq_u32(a);
  // CHECK: uaddlv {{d[0-9]+}}, {{v[0-9]+}}.4s
}

int8_t test_vmaxv_s8(int8x8_t a) {
  // CHECK-LABEL: test_vmaxv_s8
  return vmaxv_s8(a);
  // CHECK: smaxv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

int16_t test_vmaxv_s16(int16x4_t a) {
  // CHECK-LABEL: test_vmaxv_s16
  return vmaxv_s16(a);
  // CHECK: smaxv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

uint8_t test_vmaxv_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vmaxv_u8
  return vmaxv_u8(a);
  // CHECK: umaxv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

uint16_t test_vmaxv_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vmaxv_u16
  return vmaxv_u16(a);
  // CHECK: umaxv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

int8_t test_vmaxvq_s8(int8x16_t a) {
  // CHECK-LABEL: test_vmaxvq_s8
  return vmaxvq_s8(a);
  // CHECK: smaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

int16_t test_vmaxvq_s16(int16x8_t a) {
  // CHECK-LABEL: test_vmaxvq_s16
  return vmaxvq_s16(a);
  // CHECK: smaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

int32_t test_vmaxvq_s32(int32x4_t a) {
  // CHECK-LABEL: test_vmaxvq_s32
  return vmaxvq_s32(a);
  // CHECK: smaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint8_t test_vmaxvq_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vmaxvq_u8
  return vmaxvq_u8(a);
  // CHECK: umaxv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

uint16_t test_vmaxvq_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vmaxvq_u16
  return vmaxvq_u16(a);
  // CHECK: umaxv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

uint32_t test_vmaxvq_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vmaxvq_u32
  return vmaxvq_u32(a);
  // CHECK: umaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

int8_t test_vminv_s8(int8x8_t a) {
  // CHECK-LABEL: test_vminv_s8
  return vminv_s8(a);
  // CHECK: sminv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

int16_t test_vminv_s16(int16x4_t a) {
  // CHECK-LABEL: test_vminv_s16
  return vminv_s16(a);
  // CHECK: sminv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

uint8_t test_vminv_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vminv_u8
  return vminv_u8(a);
  // CHECK: uminv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

uint16_t test_vminv_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vminv_u16
  return vminv_u16(a);
  // CHECK: uminv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

int8_t test_vminvq_s8(int8x16_t a) {
  // CHECK-LABEL: test_vminvq_s8
  return vminvq_s8(a);
  // CHECK: sminv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

int16_t test_vminvq_s16(int16x8_t a) {
  // CHECK-LABEL: test_vminvq_s16
  return vminvq_s16(a);
  // CHECK: sminv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

int32_t test_vminvq_s32(int32x4_t a) {
  // CHECK-LABEL: test_vminvq_s32
  return vminvq_s32(a);
  // CHECK: sminv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint8_t test_vminvq_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vminvq_u8
  return vminvq_u8(a);
  // CHECK: uminv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

uint16_t test_vminvq_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vminvq_u16
  return vminvq_u16(a);
  // CHECK: uminv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

uint32_t test_vminvq_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vminvq_u32
  return vminvq_u32(a);
  // CHECK: uminv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

int8_t test_vaddv_s8(int8x8_t a) {
  // CHECK-LABEL: test_vaddv_s8
  return vaddv_s8(a);
  // CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

int16_t test_vaddv_s16(int16x4_t a) {
  // CHECK-LABEL: test_vaddv_s16
  return vaddv_s16(a);
  // CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

uint8_t test_vaddv_u8(uint8x8_t a) {
  // CHECK-LABEL: test_vaddv_u8
  return vaddv_u8(a);
  // CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.8b
}

uint16_t test_vaddv_u16(uint16x4_t a) {
  // CHECK-LABEL: test_vaddv_u16
  return vaddv_u16(a);
  // CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.4h
}

int8_t test_vaddvq_s8(int8x16_t a) {
  // CHECK-LABEL: test_vaddvq_s8
  return vaddvq_s8(a);
  // CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

int16_t test_vaddvq_s16(int16x8_t a) {
  // CHECK-LABEL: test_vaddvq_s16
  return vaddvq_s16(a);
  // CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

int32_t test_vaddvq_s32(int32x4_t a) {
  // CHECK-LABEL: test_vaddvq_s32
  return vaddvq_s32(a);
  // CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

uint8_t test_vaddvq_u8(uint8x16_t a) {
  // CHECK-LABEL: test_vaddvq_u8
  return vaddvq_u8(a);
  // CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.16b
}

uint16_t test_vaddvq_u16(uint16x8_t a) {
  // CHECK-LABEL: test_vaddvq_u16
  return vaddvq_u16(a);
  // CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.8h
}

uint32_t test_vaddvq_u32(uint32x4_t a) {
  // CHECK-LABEL: test_vaddvq_u32
  return vaddvq_u32(a);
  // CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

float32_t test_vmaxvq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vmaxvq_f32
  return vmaxvq_f32(a);
  // CHECK: fmaxv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

float32_t test_vminvq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vminvq_f32
  return vminvq_f32(a);
  // CHECK: fminv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

float32_t test_vmaxnmvq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vmaxnmvq_f32
  return vmaxnmvq_f32(a);
  // CHECK: fmaxnmv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

float32_t test_vminnmvq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vminnmvq_f32
  return vminnmvq_f32(a);
  // CHECK: fminnmv {{s[0-9]+}}, {{v[0-9]+}}.4s
}

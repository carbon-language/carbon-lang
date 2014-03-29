// REQUIRES: aarch64-registered-target
// REQUIRES: arm64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -S -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vand_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vand_s8
  return vand_s8(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vandq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vandq_s8
  return vandq_s8(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vand_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vand_s16
  return vand_s16(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x8_t test_vandq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vandq_s16
  return vandq_s16(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x2_t test_vand_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vand_s32
  return vand_s32(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vandq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vandq_s32
  return vandq_s32(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x1_t test_vand_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vand_s64
  return vand_s64(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int64x2_t test_vandq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vandq_s64
  return vandq_s64(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x8_t test_vand_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vand_u8
  return vand_u8(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vandq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vandq_u8
  return vandq_u8(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vand_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vand_u16
  return vand_u16(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x8_t test_vandq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vandq_u16
  return vandq_u16(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x2_t test_vand_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vand_u32
  return vand_u32(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vandq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vandq_u32
  return vandq_u32(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x1_t test_vand_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vand_u64
  return vand_u64(a, b);
  // CHECK: and {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x2_t test_vandq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vandq_u64
  return vandq_u64(a, b);
  // CHECK: and {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int8x8_t test_vorr_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vorr_s8
  return vorr_s8(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vorrq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vorrq_s8
  return vorrq_s8(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vorr_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vorr_s16
  return vorr_s16(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x8_t test_vorrq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vorrq_s16
  return vorrq_s16(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x2_t test_vorr_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vorr_s32
  return vorr_s32(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vorrq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vorrq_s32
  return vorrq_s32(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x1_t test_vorr_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vorr_s64
  return vorr_s64(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int64x2_t test_vorrq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vorrq_s64
  return vorrq_s64(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x8_t test_vorr_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vorr_u8
  return vorr_u8(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vorrq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vorrq_u8
  return vorrq_u8(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vorr_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vorr_u16
  return vorr_u16(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x8_t test_vorrq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vorrq_u16
  return vorrq_u16(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x2_t test_vorr_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vorr_u32
  return vorr_u32(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vorrq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vorrq_u32
  return vorrq_u32(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x1_t test_vorr_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vorr_u64
  return vorr_u64(a, b);
  // CHECK: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x2_t test_vorrq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vorrq_u64
  return vorrq_u64(a, b);
  // CHECK: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int8x8_t test_veor_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_veor_s8
  return veor_s8(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_veorq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_veorq_s8
  return veorq_s8(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_veor_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_veor_s16
  return veor_s16(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x8_t test_veorq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_veorq_s16
  return veorq_s16(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x2_t test_veor_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_veor_s32
  return veor_s32(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_veorq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_veorq_s32
  return veorq_s32(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x1_t test_veor_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_veor_s64
  return veor_s64(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int64x2_t test_veorq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_veorq_s64
  return veorq_s64(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x8_t test_veor_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_veor_u8
  return veor_u8(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_veorq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_veorq_u8
  return veorq_u8(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_veor_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_veor_u16
  return veor_u16(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x8_t test_veorq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_veorq_u16
  return veorq_u16(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x2_t test_veor_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_veor_u32
  return veor_u32(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_veorq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_veorq_u32
  return veorq_u32(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x1_t test_veor_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_veor_u64
  return veor_u64(a, b);
  // CHECK: eor {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x2_t test_veorq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_veorq_u64
  return veorq_u64(a, b);
  // CHECK: eor {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int8x8_t test_vbic_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vbic_s8
  return vbic_s8(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vbicq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vbicq_s8
  return vbicq_s8(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vbic_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vbic_s16
  return vbic_s16(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x8_t test_vbicq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vbicq_s16
  return vbicq_s16(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x2_t test_vbic_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vbic_s32
  return vbic_s32(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vbicq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vbicq_s32
  return vbicq_s32(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x1_t test_vbic_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vbic_s64
  return vbic_s64(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int64x2_t test_vbicq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vbicq_s64
  return vbicq_s64(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x8_t test_vbic_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vbic_u8
  return vbic_u8(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vbicq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vbicq_u8
  return vbicq_u8(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vbic_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vbic_u16
  return vbic_u16(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x8_t test_vbicq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vbicq_u16
  return vbicq_u16(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x2_t test_vbic_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vbic_u32
  return vbic_u32(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vbicq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vbicq_u32
  return vbicq_u32(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x1_t test_vbic_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vbic_u64
  return vbic_u64(a, b);
  // CHECK: bic {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x2_t test_vbicq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vbicq_u64
  return vbicq_u64(a, b);
  // CHECK: bic {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int8x8_t test_vorn_s8(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_vorn_s8
  return vorn_s8(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vornq_s8(int8x16_t a, int8x16_t b) {
  // CHECK-LABEL: test_vornq_s8
  return vornq_s8(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x4_t test_vorn_s16(int16x4_t a, int16x4_t b) {
  // CHECK-LABEL: test_vorn_s16
  return vorn_s16(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x8_t test_vornq_s16(int16x8_t a, int16x8_t b) {
  // CHECK-LABEL: test_vornq_s16
  return vornq_s16(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x2_t test_vorn_s32(int32x2_t a, int32x2_t b) {
  // CHECK-LABEL: test_vorn_s32
  return vorn_s32(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x4_t test_vornq_s32(int32x4_t a, int32x4_t b) {
  // CHECK-LABEL: test_vornq_s32
  return vornq_s32(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x1_t test_vorn_s64(int64x1_t a, int64x1_t b) {
  // CHECK-LABEL: test_vorn_s64
  return vorn_s64(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int64x2_t test_vornq_s64(int64x2_t a, int64x2_t b) {
  // CHECK-LABEL: test_vornq_s64
  return vornq_s64(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x8_t test_vorn_u8(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_vorn_u8
  return vorn_u8(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vornq_u8(uint8x16_t a, uint8x16_t b) {
  // CHECK-LABEL: test_vornq_u8
  return vornq_u8(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x4_t test_vorn_u16(uint16x4_t a, uint16x4_t b) {
  // CHECK-LABEL: test_vorn_u16
  return vorn_u16(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x8_t test_vornq_u16(uint16x8_t a, uint16x8_t b) {
  // CHECK-LABEL: test_vornq_u16
  return vornq_u16(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint32x2_t test_vorn_u32(uint32x2_t a, uint32x2_t b) {
  // CHECK-LABEL: test_vorn_u32
  return vorn_u32(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x4_t test_vornq_u32(uint32x4_t a, uint32x4_t b) {
  // CHECK-LABEL: test_vornq_u32
  return vornq_u32(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x1_t test_vorn_u64(uint64x1_t a, uint64x1_t b) {
  // CHECK-LABEL: test_vorn_u64
  return vorn_u64(a, b);
  // CHECK: orn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x2_t test_vornq_u64(uint64x2_t a, uint64x2_t b) {
  // CHECK-LABEL: test_vornq_u64
  return vornq_u64(a, b);
  // CHECK: orn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

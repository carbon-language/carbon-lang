// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK_CODEGEN %s
// REQUIRES: arm64-registered-target

// Test ARM64 SIMD vector shift right and insert: vsri[q]_n_*

#include <arm_neon.h>

int8x8_t test_vsri_n_s8(int8x8_t a1, int8x8_t a2) {
  // CHECK: test_vsri_n_s8
  return vsri_n_s8(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v8i8
  // CHECK_CODEGEN: sri.8b  v0, v1, #3
}

int16x4_t test_vsri_n_s16(int16x4_t a1, int16x4_t a2) {
  // CHECK: test_vsri_n_s16
  return vsri_n_s16(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v4i16
  // CHECK_CODEGEN: sri.4h  v0, v1, #3
}

int32x2_t test_vsri_n_s32(int32x2_t a1, int32x2_t a2) {
  // CHECK: test_vsri_n_s32
  return vsri_n_s32(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v2i32
  // CHECK_CODEGEN: sri.2s  v0, v1, #1
}

int64x1_t test_vsri_n_s64(int64x1_t a1, int64x1_t a2) {
  // CHECK: test_vsri_n_s64
  return vsri_n_s64(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v1i64
  // CHECK_CODEGEN: sri     d0, d1, #1
}

uint8x8_t test_vsri_n_u8(uint8x8_t a1, uint8x8_t a2) {
  // CHECK: test_vsri_n_u8
  return vsri_n_u8(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v8i8
  // CHECK_CODEGEN: sri.8b  v0, v1, #3
}

uint16x4_t test_vsri_n_u16(uint16x4_t a1, uint16x4_t a2) {
  // CHECK: test_vsri_n_u16
  return vsri_n_u16(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v4i16
  // CHECK_CODEGEN: sri.4h  v0, v1, #3
}

uint32x2_t test_vsri_n_u32(uint32x2_t a1, uint32x2_t a2) {
  // CHECK: test_vsri_n_u32
  return vsri_n_u32(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v2i32
  // CHECK_CODEGEN: sri.2s  v0, v1, #1
}

uint64x1_t test_vsri_n_u64(uint64x1_t a1, uint64x1_t a2) {
  // CHECK: test_vsri_n_u64
  return vsri_n_u64(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v1i64
  // CHECK_CODEGEN: sri     d0, d1, #1
}

poly8x8_t test_vsri_n_p8(poly8x8_t a1, poly8x8_t a2) {
  // CHECK: test_vsri_n_p8
  return vsri_n_p8(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v8i8
  // CHECK_CODEGEN: sri.8b  v0, v1, #1
}

poly16x4_t test_vsri_n_p16(poly16x4_t a1, poly16x4_t a2) {
  // CHECK: test_vsri_n_p16
  return vsri_n_p16(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v4i16
  // CHECK_CODEGEN: sri.4h  v0, v1, #1
}

int8x16_t test_vsriq_n_s8(int8x16_t a1, int8x16_t a2) {
  // CHECK: test_vsriq_n_s8
  return vsriq_n_s8(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v16i8
  // CHECK_CODEGEN: sri.16b v0, v1, #3
}

int16x8_t test_vsriq_n_s16(int16x8_t a1, int16x8_t a2) {
  // CHECK: test_vsriq_n_s16
  return vsriq_n_s16(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v8i16
  // CHECK_CODEGEN: sri.8h  v0, v1, #3
}

int32x4_t test_vsriq_n_s32(int32x4_t a1, int32x4_t a2) {
  // CHECK: test_vsriq_n_s32
  return vsriq_n_s32(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v4i32
  // CHECK_CODEGEN: sri.4s  v0, v1, #1
}

int64x2_t test_vsriq_n_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vsriq_n_s64
  return vsriq_n_s64(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v2i64
  // CHECK_CODEGEN: sri.2d  v0, v1, #1
}

uint8x16_t test_vsriq_n_u8(uint8x16_t a1, uint8x16_t a2) {
  // CHECK: test_vsriq_n_u8
  return vsriq_n_u8(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v16i8
  // CHECK_CODEGEN: sri.16b v0, v1, #3
}

uint16x8_t test_vsriq_n_u16(uint16x8_t a1, uint16x8_t a2) {
  // CHECK: test_vsriq_n_u16
  return vsriq_n_u16(a1, a2, 3);
  // CHECK: llvm.aarch64.neon.vsri.v8i16
  // CHECK_CODEGEN: sri.8h  v0, v1, #3
}

uint32x4_t test_vsriq_n_u32(uint32x4_t a1, uint32x4_t a2) {
  // CHECK: test_vsriq_n_u32
  return vsriq_n_u32(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v4i32
  // CHECK_CODEGEN: sri.4s  v0, v1, #1
}

uint64x2_t test_vsriq_n_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vsriq_n_u64
  return vsriq_n_u64(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v2i64
  // CHECK_CODEGEN: sri.2d  v0, v1, #1
}

poly8x16_t test_vsriq_n_p8(poly8x16_t a1, poly8x16_t a2) {
  // CHECK: test_vsriq_n_p8
  return vsriq_n_p8(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v16i8
  // CHECK_CODEGEN: sri.16b v0, v1, #1
}

poly16x8_t test_vsriq_n_p16(poly16x8_t a1, poly16x8_t a2) {
  // CHECK: test_vsriq_n_p16
  return vsriq_n_p16(a1, a2, 1);
  // CHECK: llvm.aarch64.neon.vsri.v8i16
  // CHECK_CODEGEN: sri.8h  v0, v1, #1
}


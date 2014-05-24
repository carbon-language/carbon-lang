// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD add intrinsics

#include <arm_neon.h>
int64_t test_vaddlv_s32(int32x2_t a1) {
  // CHECK: test_vaddlv_s32
  return vaddlv_s32(a1);
  // CHECK: llvm.aarch64.neon.saddlv.i64.v2i32
  // CHECK-NEXT: ret
}

uint64_t test_vaddlv_u32(uint32x2_t a1) {
  // CHECK: test_vaddlv_u32
  return vaddlv_u32(a1);
  // CHECK: llvm.aarch64.neon.uaddlv.i64.v2i32
  // CHECK-NEXT: ret
}

int8_t test_vaddv_s8(int8x8_t a1) {
  // CHECK: test_vaddv_s8
  return vaddv_s8(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v8i8
  // don't check for return here (there's a trunc?)
}

int16_t test_vaddv_s16(int16x4_t a1) {
  // CHECK: test_vaddv_s16
  return vaddv_s16(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v4i16
  // don't check for return here (there's a trunc?)
}

int32_t test_vaddv_s32(int32x2_t a1) {
  // CHECK: test_vaddv_s32
  return vaddv_s32(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v2i32
  // CHECK-NEXT: ret
}

uint8_t test_vaddv_u8(int8x8_t a1) {
  // CHECK: test_vaddv_u8
  return vaddv_u8(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v8i8
  // don't check for return here (there's a trunc?)
}

uint16_t test_vaddv_u16(int16x4_t a1) {
  // CHECK: test_vaddv_u16
  return vaddv_u16(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v4i16
  // don't check for return here (there's a trunc?)
}

uint32_t test_vaddv_u32(int32x2_t a1) {
  // CHECK: test_vaddv_u32
  return vaddv_u32(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v2i32
  // CHECK-NEXT: ret
}

int8_t test_vaddvq_s8(int8x16_t a1) {
  // CHECK: test_vaddvq_s8
  return vaddvq_s8(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v16i8
  // don't check for return here (there's a trunc?)
}

int16_t test_vaddvq_s16(int16x8_t a1) {
  // CHECK: test_vaddvq_s16
  return vaddvq_s16(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v8i16
  // don't check for return here (there's a trunc?)
}

int32_t test_vaddvq_s32(int32x4_t a1) {
  // CHECK: test_vaddvq_s32
  return vaddvq_s32(a1);
  // CHECK: llvm.aarch64.neon.saddv.i32.v4i32
  // CHECK-NEXT: ret
}

uint8_t test_vaddvq_u8(int8x16_t a1) {
  // CHECK: test_vaddvq_u8
  return vaddvq_u8(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v16i8
  // don't check for return here (there's a trunc?)
}

uint16_t test_vaddvq_u16(int16x8_t a1) {
  // CHECK: test_vaddvq_u16
  return vaddvq_u16(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v8i16
  // don't check for return here (there's a trunc?)
}

uint32_t test_vaddvq_u32(int32x4_t a1) {
  // CHECK: test_vaddvq_u32
  return vaddvq_u32(a1);
  // CHECK: llvm.aarch64.neon.uaddv.i32.v4i32
  // CHECK-NEXT: ret
}


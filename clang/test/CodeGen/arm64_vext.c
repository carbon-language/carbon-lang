// RUN: %clang_cc1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s

// Test ARM64 extract intrinsics
// can use as back end test by adding a run line with
// -check-prefix=CHECK-CODEGEN on the FileCheck

#include <arm_neon.h>

void test_vext_s8()
{
  // CHECK: test_vext_s8
  int8x8_t xS8x8;
  xS8x8 = vext_s8(xS8x8, xS8x8, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_s8:
  // CHECK-CODEGEN: {{ext.8.*#1}}
}

void test_vext_u8()
{
  // CHECK: test_vext_u8
  uint8x8_t xU8x8;
  xU8x8 = vext_u8(xU8x8, xU8x8, 2);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_u8:
  // CHECK-CODEGEN: {{ext.8.*#2}}
}

void test_vext_p8()
{
  // CHECK: test_vext_p8
  poly8x8_t xP8x8;
  xP8x8 = vext_p8(xP8x8, xP8x8, 3);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_p8:
  // CHECK-CODEGEN: {{ext.8.*#3}}
}

void test_vext_s16()
{
  // CHECK: test_vext_s16
  int16x4_t xS16x4;
  xS16x4 = vext_s16(xS16x4, xS16x4, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_s16:
  // CHECK-CODEGEN: {{ext.8.*#2}}
}

void test_vext_u16()
{
  // CHECK: test_vext_u16
  uint16x4_t xU16x4;
  xU16x4 = vext_u16(xU16x4, xU16x4, 2);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_u16:
  // CHECK-CODEGEN: {{ext.8.*#4}}
}

void test_vext_p16()
{
  // CHECK: test_vext_p16
  poly16x4_t xP16x4;
  xP16x4 = vext_p16(xP16x4, xP16x4, 3);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_p16:
  // CHECK-CODEGEN: {{ext.8.*#6}}
}

void test_vext_s32()
{
  // CHECK: test_vext_s32
  int32x2_t xS32x2;
  xS32x2 = vext_s32(xS32x2, xS32x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_s32:
  // CHECK-CODEGEN: {{ext.8.*#4}}
}

void test_vext_u32()
{
  // CHECK: test_vext_u32
  uint32x2_t xU32x2;
  xU32x2 = vext_u32(xU32x2, xU32x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_u32:
  // CHECK-CODEGEN: {{ext.8.*#4}}
}

void test_vext_f32()
{
  // CHECK: test_vext_f32
  float32x2_t xF32x2;
  xF32x2 = vext_f32(xF32x2, xF32x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_f32:
  // CHECK-CODEGEN: {{ext.8.*#4}}
}

void test_vext_s64()
{
  // CHECK: test_vext_s64
  int64x1_t xS64x1;
  // FIXME don't use 1 as index or check for now, clang has a bug?
  xS64x1 = vext_s64(xS64x1, xS64x1, /*1*/0);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_s64:
  // CHECK_FIXME: {{ext.8.*#0}}
}

void test_vext_u64()
{
  // CHECK: test_vext_u64
  uint64x1_t xU64x1;
  // FIXME don't use 1 as index or check for now, clang has a bug?
  xU64x1 = vext_u64(xU64x1, xU64x1, /*1*/0);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vext_u64:
  // CHECK_FIXME: {{ext.8.*#0}}
}

void test_vextq_s8()
{
  // CHECK: test_vextq_s8
  int8x16_t xS8x16;
  xS8x16 = vextq_s8(xS8x16, xS8x16, 4);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_s8:
  // CHECK-CODEGEN: {{ext.16.*#4}}
}

void test_vextq_u8()
{
  // CHECK: test_vextq_u8
  uint8x16_t xU8x16;
  xU8x16 = vextq_u8(xU8x16, xU8x16, 5);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_u8:
  // CHECK-CODEGEN: {{ext.16.*#5}}
}

void test_vextq_p8()
{
  // CHECK: test_vextq_p8
  poly8x16_t xP8x16;
  xP8x16 = vextq_p8(xP8x16, xP8x16, 6);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_p8:
  // CHECK-CODEGEN: {{ext.16.*#6}}
}

void test_vextq_s16()
{
  // CHECK: test_vextq_s16
  int16x8_t xS16x8;
  xS16x8 = vextq_s16(xS16x8, xS16x8, 7);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_s16:
  // CHECK-CODEGEN: {{ext.16.*#14}}
}

void test_vextq_u16()
{
  // CHECK: test_vextq_u16
  uint16x8_t xU16x8;
  xU16x8 = vextq_u16(xU16x8, xU16x8, 4);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_u16:
  // CHECK-CODEGEN: {{ext.16.*#8}}
}

void test_vextq_p16()
{
  // CHECK: test_vextq_p16
  poly16x8_t xP16x8;
  xP16x8 = vextq_p16(xP16x8, xP16x8, 5);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_p16:
  // CHECK-CODEGEN: {{ext.16.*#10}}
}

void test_vextq_s32()
{
  // CHECK: test_vextq_s32
  int32x4_t xS32x4;
  xS32x4 = vextq_s32(xS32x4, xS32x4, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_s32:
  // CHECK-CODEGEN: {{ext.16.*#4}}
}

void test_vextq_u32()
{
  // CHECK: test_vextq_u32
  uint32x4_t xU32x4;
  xU32x4 = vextq_u32(xU32x4, xU32x4, 2);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_u32:
  // CHECK-CODEGEN: {{ext.16.*#8}}
}

void test_vextq_f32()
{
  // CHECK: test_vextq_f32
  float32x4_t xF32x4;
  xF32x4 = vextq_f32(xF32x4, xF32x4, 3);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_f32:
  // CHECK-CODEGEN: {{ext.16.*#12}}
}

void test_vextq_s64()
{
  // CHECK: test_vextq_s64
  int64x2_t xS64x2;
  xS64x2 = vextq_s64(xS64x2, xS64x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_s64:
  // CHECK-CODEGEN: {{ext.16.*#8}}
}

void test_vextq_u64()
{
  // CHECK: test_vextq_u64
  uint64x2_t xU64x2;
  xU64x2 = vextq_u64(xU64x2, xU64x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_u64:
  // CHECK-CODEGEN: {{ext.16.*#8}}
}

void test_vextq_f64()
{
  // CHECK: test_vextq_f64
  float64x2_t xF64x2;
  xF64x2 = vextq_f64(xF64x2, xF64x2, 1);
  // CHECK: shufflevector
  // CHECK-CODEGEN: test_vextq_u64:
  // CHECK-CODEGEN: {{ext.16.*#8}}
}

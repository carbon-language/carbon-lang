// RUN: %clang_cc1 -O3 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -O3 -triple aarch64_be-linux-gnu -target-feature +neon -ffreestanding -emit-llvm -o - %s | FileCheck %s --check-prefix CHECK-BE

#include <arm_neon.h>

// CHECK-LABEL: @test_vdupb_lane_s8
int8_t test_vdupb_lane_s8(int8x8_t src) {
  return vdupb_lane_s8(src, 2);
  // CHECK: extractelement <8 x i8> %src, i32 2
  // CHECK-BE: extractelement <8 x i8> %src, i32 5
}

// CHECK-LABEL: @test_vdupb_lane_u8
uint8_t test_vdupb_lane_u8(uint8x8_t src) {
  return vdupb_lane_u8(src, 2);
  // CHECK: extractelement <8 x i8> %src, i32 2
  // CHECK-BE: extractelement <8 x i8> %src, i32 5
}

// CHECK-LABEL: @test_vduph_lane_s16
int16_t test_vduph_lane_s16(int16x4_t src) {
  return vduph_lane_s16(src, 2);
  // CHECK: extractelement <4 x i16> %src, i32 2
  // CHECK-BE: extractelement <4 x i16> %src, i32 1
}

// CHECK-LABEL: @test_vduph_lane_u16
uint16_t test_vduph_lane_u16(uint16x4_t src) {
  return vduph_lane_u16(src, 2);
  // CHECK: extractelement <4 x i16> %src, i32 2
  // CHECK-BE: extractelement <4 x i16> %src, i32 1
}

// CHECK-LABEL: @test_vdups_lane_s32
int32_t test_vdups_lane_s32(int32x2_t src) {
  return vdups_lane_s32(src, 0);
  // CHECK: extractelement <2 x i32> %src, i32 0
  // CHECK-BE: extractelement <2 x i32> %src, i32 1
}

// CHECK-LABEL: @test_vdups_lane_u32
uint32_t test_vdups_lane_u32(uint32x2_t src) {
  return vdups_lane_u32(src, 0);
  // CHECK: extractelement <2 x i32> %src, i32 0
  // CHECK-BE: extractelement <2 x i32> %src, i32 1
}

// CHECK-LABEL: @test_vdups_lane_f32
float32_t test_vdups_lane_f32(float32x2_t src) {
  return vdups_lane_f32(src, 0);
  // CHECK: extractelement <2 x float> %src, i32 0
  // CHECK-BE: extractelement <2 x float> %src, i32 1
}

// CHECK-LABEL: @test_vdupd_lane_s64
int64_t test_vdupd_lane_s64(int64x1_t src) {
  return vdupd_lane_s64(src, 0);
  // CHECK: extractelement <1 x i64> %src, i32 0
  // CHECK-BE: extractelement <1 x i64> %src, i32 0
}

// CHECK-LABEL: @test_vdupd_lane_u64
uint64_t test_vdupd_lane_u64(uint64x1_t src) {
  return vdupd_lane_u64(src, 0);
  // CHECK: extractelement <1 x i64> %src, i32 0
  // CHECK-BE: extractelement <1 x i64> %src, i32 0
}

// CHECK-LABEL: @test_vdupd_lane_f64
float64_t test_vdupd_lane_f64(float64x1_t src) {
  return vdupd_lane_f64(src, 0);
  // CHECK: extractelement <1 x double> %src, i32 0
  // CHECK-BE: extractelement <1 x double> %src, i32 0
}

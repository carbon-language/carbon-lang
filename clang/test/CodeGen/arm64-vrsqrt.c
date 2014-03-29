// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -ffreestanding -emit-llvm -O1 -o - %s | FileCheck %s

#include <arm_neon.h>

uint32x2_t test_vrsqrte_u32(uint32x2_t in) {
  // CHECK-LABEL: @test_vrsqrte_u32
  // CHECK: call <2 x i32> @llvm.arm64.neon.ursqrte.v2i32(<2 x i32> %in)
  return vrsqrte_u32(in);
}

float32x2_t test_vrsqrte_f32(float32x2_t in) {
  // CHECK-LABEL: @test_vrsqrte_f32
  // CHECK: call <2 x float> @llvm.arm64.neon.frsqrte.v2f32(<2 x float> %in)
  return vrsqrte_f32(in);
}


uint32x4_t test_vrsqrteq_u32(uint32x4_t in) {
  // CHECK-LABEL: @test_vrsqrteq_u32
  // CHECK: call <4 x i32> @llvm.arm64.neon.ursqrte.v4i32(<4 x i32> %in)
  return vrsqrteq_u32(in);
}

float32x4_t test_vrsqrteq_f32(float32x4_t in) {
  // CHECK-LABEL: @test_vrsqrteq_f32
  // CHECK: call <4 x float> @llvm.arm64.neon.frsqrte.v4f32(<4 x float> %in)
  return vrsqrteq_f32(in);
}


float32x2_t test_vrsqrts_f32(float32x2_t est, float32x2_t val) {
  // CHECK-LABEL: @test_vrsqrts_f32
  // CHECK: call <2 x float> @llvm.arm64.neon.frsqrts.v2f32(<2 x float> %est, <2 x float> %val)
  return vrsqrts_f32(est, val);
}


float32x4_t test_vrsqrtsq_f32(float32x4_t est, float32x4_t val) {
  // CHECK-LABEL: @test_vrsqrtsq_f32
  // CHECK: call <4 x float> @llvm.arm64.neon.frsqrts.v4f32(<4 x float> %est, <4 x float> %val)
  return vrsqrtsq_f32(est, val);
}


// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7-none-linux-gnueabihf \
// RUN:   -target-abi aapcs \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi hard \
// RUN:   -ffreestanding \
// RUN:   -O3 -S -emit-llvm -o - %s | FileCheck %s

#include <arm_neon.h>

float32x2_t test_fma_order(float32x2_t accum, float32x2_t lhs, float32x2_t rhs) {
  return vfma_f32(accum, lhs, rhs);
// CHECK: call <2 x float> @llvm.fma.v2f32(<2 x float> %lhs, <2 x float> %rhs, <2 x float> %accum)
}

float32x4_t test_fmaq_order(float32x4_t accum, float32x4_t lhs, float32x4_t rhs) {
  return vfmaq_f32(accum, lhs, rhs);
// CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %lhs, <4 x float> %rhs, <4 x float> %accum)
}

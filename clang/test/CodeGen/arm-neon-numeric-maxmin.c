// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -O1 -emit-llvm %s -o - | FileCheck %s

#include <arm_neon.h>

float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vmaxnm_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float> %a, <2 x float> %b)
  return vmaxnm_f32(a, b);
}

float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vmaxnmq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float> %a, <4 x float> %b)
  return vmaxnmq_f32(a, b);
}

float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
  // CHECK-LABEL: test_vminnm_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float> %a, <2 x float> %b)
  return vminnm_f32(a, b);
}

float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
  // CHECK-LABEL: test_vminnmq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float> %a, <4 x float> %b)
  return vminnmq_f32(a, b);
}

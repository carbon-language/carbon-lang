// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -O1 -emit-llvm %s -o - | FileCheck %s

#include <arm_neon.h>

float32x2_t test_vrnda_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrnda_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrinta.v2f32(<2 x float> %a)
  return vrnda_f32(a);
}

float32x4_t test_vrndaq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndaq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrinta.v4f32(<4 x float> %a)
  return vrndaq_f32(a);
}

float32x2_t test_vrndm_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrndm_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrintm.v2f32(<2 x float> %a)
  return vrndm_f32(a);
}

float32x4_t test_vrndmq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndmq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrintm.v4f32(<4 x float> %a)
  return vrndmq_f32(a);
}

float32x2_t test_vrndn_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrndn_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float> %a)
  return vrndn_f32(a);
}

float32x4_t test_vrndnq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndnq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float> %a)
  return vrndnq_f32(a);
}

float32x2_t test_vrndp_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrndp_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrintp.v2f32(<2 x float> %a)
  return vrndp_f32(a);
}

float32x4_t test_vrndpq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndpq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrintp.v4f32(<4 x float> %a)
  return vrndpq_f32(a);
}

float32x2_t test_vrndx_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrndx_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrintx.v2f32(<2 x float> %a)
  return vrndx_f32(a);
}

float32x4_t test_vrndxq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndxq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrintx.v4f32(<4 x float> %a)
  return vrndxq_f32(a);
}

float32x2_t test_vrnd_f32(float32x2_t a) {
  // CHECK-LABEL: test_vrnd_f32
  // CHECK: call <2 x float> @llvm.arm.neon.vrintz.v2f32(<2 x float> %a)
  return vrnd_f32(a);
}

float32x4_t test_vrndq_f32(float32x4_t a) {
  // CHECK-LABEL: test_vrndq_f32
  // CHECK: call <4 x float> @llvm.arm.neon.vrintz.v4f32(<4 x float> %a)
  return vrndq_f32(a);
}

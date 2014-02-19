// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -O1 -emit-llvm %s -o - | FileCheck %s

#include <arm_neon.h>

int32x2_t test_vcvta_s32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvta_s32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtas.v2i32.v2f32(<2 x float> %a)
  return vcvta_s32_f32(a);
}

uint32x2_t test_vcvta_u32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvta_u32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtau.v2i32.v2f32(<2 x float> %a)
  return vcvta_u32_f32(a);
}

int32x4_t test_vcvtaq_s32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtaq_s32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtas.v4i32.v4f32(<4 x float> %a)
  return vcvtaq_s32_f32(a);
}

uint32x4_t test_vcvtaq_u32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtaq_u32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtau.v4i32.v4f32(<4 x float> %a)
  return vcvtaq_u32_f32(a);
}

int32x2_t test_vcvtn_s32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtn_s32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtns.v2i32.v2f32(<2 x float> %a)
  return vcvtn_s32_f32(a);
}

uint32x2_t test_vcvtn_u32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtn_u32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtnu.v2i32.v2f32(<2 x float> %a)
  return vcvtn_u32_f32(a);
}

int32x4_t test_vcvtnq_s32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtnq_s32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtns.v4i32.v4f32(<4 x float> %a)
  return vcvtnq_s32_f32(a);
}

uint32x4_t test_vcvtnq_u32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtnq_u32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtnu.v4i32.v4f32(<4 x float> %a)
  return vcvtnq_u32_f32(a);
}

int32x2_t test_vcvtp_s32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtp_s32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtps.v2i32.v2f32(<2 x float> %a)
  return vcvtp_s32_f32(a);
}

uint32x2_t test_vcvtp_u32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtp_u32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtpu.v2i32.v2f32(<2 x float> %a)
  return vcvtp_u32_f32(a);
}

int32x4_t test_vcvtpq_s32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtpq_s32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtps.v4i32.v4f32(<4 x float> %a)
  return vcvtpq_s32_f32(a);
}

uint32x4_t test_vcvtpq_u32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtpq_u32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtpu.v4i32.v4f32(<4 x float> %a)
  return vcvtpq_u32_f32(a);
}

int32x2_t test_vcvtm_s32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtm_s32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtms.v2i32.v2f32(<2 x float> %a)
  return vcvtm_s32_f32(a);
}

uint32x2_t test_vcvtm_u32_f32(float32x2_t a) {
  // CHECK-LABEL: test_vcvtm_u32_f32
  // CHECK-LABEL: call <2 x i32> @llvm.arm.neon.vcvtmu.v2i32.v2f32(<2 x float> %a)
  return vcvtm_u32_f32(a);
}

int32x4_t test_vcvtmq_s32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtmq_s32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtms.v4i32.v4f32(<4 x float> %a)
  return vcvtmq_s32_f32(a);
}

uint32x4_t test_vcvtmq_u32_f32(float32x4_t a) {
  // CHECK-LABEL: test_vcvtmq_u32_f32
  // CHECK-LABEL: call <4 x i32> @llvm.arm.neon.vcvtmu.v4i32.v4f32(<4 x float> %a)
  return vcvtmq_u32_f32(a);
}

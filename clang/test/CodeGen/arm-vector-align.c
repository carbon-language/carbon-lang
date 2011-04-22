// RUN: %clang_cc1 -triple thumbv7-apple-darwin \
// RUN:   -target-abi apcs-gnu \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi soft \
// RUN:   -target-feature +soft-float-abi \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

#include <arm_neon.h>

// Radar 9311427: Check that alignment specifier is used in Neon load/store
// intrinsics.
typedef float AlignedAddr __attribute__ ((aligned (16)));
void t1(AlignedAddr *addr1, AlignedAddr *addr2) {
// CHECK: call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %{{.*}}, i32 16)
  float32x4_t a = vld1q_f32(addr1);
// CHECK: call void @llvm.arm.neon.vst1.v4f32(i8* %{{.*}}, <4 x float> %{{.*}}, i32 16)
  vst1q_f32(addr2, a);
}

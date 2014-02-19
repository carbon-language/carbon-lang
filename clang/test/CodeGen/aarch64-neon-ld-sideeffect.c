// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -emit-llvm -O0 -o - %s | FileCheck %s

#include <arm_neon.h>

void *foo(void);

float32x2_t bar(void) {
  // CHECK-LABEL: @bar
  return vld1_f32(foo());
  // CHECK: call i8* @foo
  // CHECK-NOT: call i8* @foo
  // CHECK: call <2 x float> @llvm.arm.neon.vld1
}

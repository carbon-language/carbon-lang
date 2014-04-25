// RUN: %clang_cc1 -triple thumbv7-windows -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-eabi -emit-llvm -o - %s | FileCheck %s
// REQUIRES: arm-registered-target

void test_yield_intrinsic() {
  __yield();
}

// CHECK: call void @llvm.arm.hint(i32 1)


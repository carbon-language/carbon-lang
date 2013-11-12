// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -Wall -Werror -triple thumbv7-eabi -target-cpu cortex-a8 -O3 -emit-llvm -o - %s | FileCheck %s

void *f0()
{
  return __builtin_thread_pointer();
}

void f1(char *a, char *b) {
	__clear_cache(a,b);
}

// CHECK: call {{.*}} @__clear_cache

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 0
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 1
}

void sevl() {
  __builtin_arm_sevl();
}
// CHECK: call {{.*}} @llvm.arm.sevl

void test_barrier() {
  __builtin_arm_dmb(1); //CHECK: call {{.*}} @llvm.arm.dmb(i32 1)
  __builtin_arm_dsb(2); //CHECK: call {{.*}} @llvm.arm.dsb(i32 2)
}

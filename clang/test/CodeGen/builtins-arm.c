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

void yield() {
  __builtin_arm_yield();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 1)

void wfe() {
  __builtin_arm_wfe();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 2)

void wfi() {
  __builtin_arm_wfi();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 3)

void sev() {
  __builtin_arm_sev();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 4)

void sevl() {
  __builtin_arm_sevl();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 5)

void test_barrier() {
  __builtin_arm_dmb(1); //CHECK: call {{.*}} @llvm.arm.dmb(i32 1)
  __builtin_arm_dsb(2); //CHECK: call {{.*}} @llvm.arm.dsb(i32 2)
  __builtin_arm_isb(3); //CHECK: call {{.*}} @llvm.arm.isb(i32 3)
}

// CHECK: call {{.*}} @llvm.arm.rbit(i32 %a)

unsigned rbit(unsigned a) {
  return __builtin_arm_rbit(a);
}

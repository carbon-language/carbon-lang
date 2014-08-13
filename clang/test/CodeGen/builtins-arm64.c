// RUN: %clang_cc1 -triple arm64-apple-ios -O3 -emit-llvm -o - %s | FileCheck %s

void f0(void *a, void *b) {
	__clear_cache(a,b);
// CHECK: call {{.*}} @__clear_cache
}

// CHECK: call {{.*}} @llvm.aarch64.rbit.i32(i32 %a)
unsigned rbit(unsigned a) {
  return __builtin_arm_rbit(a);
}

// CHECK: call {{.*}} @llvm.aarch64.rbit.i64(i64 %a)
unsigned long long rbit64(unsigned long long a) {
  return __builtin_arm_rbit64(a);
}

void hints() {
  __builtin_arm_nop();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 0)
  __builtin_arm_yield();  //CHECK: call {{.*}} @llvm.aarch64.hint(i32 1)
  __builtin_arm_wfe();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 2)
  __builtin_arm_wfi();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 3)
  __builtin_arm_sev();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 4)
  __builtin_arm_sevl();   //CHECK: call {{.*}} @llvm.aarch64.hint(i32 5)
}

void barriers() {
  __builtin_arm_dmb(1);  //CHECK: call {{.*}} @llvm.aarch64.dmb(i32 1)
  __builtin_arm_dsb(2);  //CHECK: call {{.*}} @llvm.aarch64.dsb(i32 2)
  __builtin_arm_isb(3);  //CHECK: call {{.*}} @llvm.aarch64.isb(i32 3)
}

void prefetch() {
  __builtin_arm_prefetch(0, 1, 2, 0, 1); // pstl3keep
// CHECK: call {{.*}} @llvm.prefetch(i8* null, i32 1, i32 1, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 1, 1); // pldl1keep
// CHECK: call {{.*}} @llvm.prefetch(i8* null, i32 0, i32 0, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 1, 1); // pldl1strm
// CHECK: call {{.*}} @llvm.prefetch(i8* null, i32 0, i32 0, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 0, 0); // plil1keep
// CHECK: call {{.*}} @llvm.prefetch(i8* null, i32 0, i32 3, i32 0)
}

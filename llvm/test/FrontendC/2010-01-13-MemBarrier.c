// RUN: %llvmgcc %s -S -emit-llvm -o - | FileCheck %s
// XFAIL: sparc
// rdar://7536390

unsigned t(unsigned *ptr, unsigned val) {
  // CHECK:      @t
  // CHECK:      call void @llvm.memory.barrier
  // CHECK-NEXT: call i32 @llvm.atomic.swap.i32
  // CHECK-NEXT: call void @llvm.memory.barrier
  return __sync_lock_test_and_set(ptr, val);
}

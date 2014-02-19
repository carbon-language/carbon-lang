// RUN: %clang_cc1 -triple=i686-unknown-unknown -fms-extensions -emit-llvm -o - %s | FileCheck %s

extern void printf(const char*, ...);
void f(char *a, volatile long* b) {
  _mm_prefetch(a, 0);
  _mm_prefetch(a, 1);
  _mm_prefetch(a, 2);
  _mm_prefetch(a, 3);
  _InterlockedCompareExchange(b, 1, 0);
  _InterlockedIncrement(b);
  _InterlockedDecrement(b);
  _InterlockedExchangeAdd(b, 2);
};

// CHECK: call void @llvm.prefetch(i8* %1, i32 0, i32 0, i32 1)
// CHECK: call void @llvm.prefetch(i8* %3, i32 0, i32 1, i32 1)
// CHECK: call void @llvm.prefetch(i8* %5, i32 0, i32 2, i32 1)
// CHECK: call void @llvm.prefetch(i8* %7, i32 0, i32 3, i32 1)
// CHECK: cmpxchg
// CHECK: atomicrmw volatile add
// CHECK: atomicrmw volatile sub
// CHECK: atomicrmw volatile add

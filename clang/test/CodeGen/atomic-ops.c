// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

// Basic IRGen tests for __atomic_*

// FIXME: Need to implement __atomic_is_lock_free

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

int fi1(_Atomic(int) *i) {
  // CHECK: @fi1
  // CHECK: load atomic i32* {{.*}} seq_cst
  return __atomic_load(i, memory_order_seq_cst);
}

void fi2(_Atomic(int) *i) {
  // CHECK: @fi2
  // CHECK: store atomic i32 {{.*}} seq_cst
  __atomic_store(i, 1, memory_order_seq_cst);
}

void fi3(_Atomic(int) *i) {
  // CHECK: @fi3
  // CHECK: atomicrmw and
  __atomic_fetch_and(i, 1, memory_order_seq_cst);
}

void fi4(_Atomic(int) *i) {
  // CHECK: @fi4
  // CHECK: cmpxchg i32*
  int cmp = 0;
  __atomic_compare_exchange_strong(i, &cmp, 1, memory_order_acquire, memory_order_acquire);
}

float ff1(_Atomic(float) *d) {
  // CHECK: @ff1
  // CHECK: load atomic i32* {{.*}} monotonic
  return __atomic_load(d, memory_order_relaxed);
}

void ff2(_Atomic(float) *d) {
  // CHECK: @ff2
  // CHECK: store atomic i32 {{.*}} release
  __atomic_store(d, 1, memory_order_release);
}

float ff3(_Atomic(float) *d) {
  return __atomic_exchange(d, 2, memory_order_seq_cst);
}

int* fp1(_Atomic(int*) *p) {
  // CHECK: @fp1
  // CHECK: load atomic i32* {{.*}} seq_cst
  return __atomic_load(p, memory_order_seq_cst);
}

int* fp2(_Atomic(int*) *p) {
  // CHECK: @fp2
  // CHECK: store i32 4
  // CHECK: atomicrmw add {{.*}} monotonic
  return __atomic_fetch_add(p, 1, memory_order_relaxed);
}

_Complex float fc(_Atomic(_Complex float) *c) {
  // CHECK: @fc
  // CHECK: atomicrmw xchg i64*
  return __atomic_exchange(c, 2, memory_order_seq_cst);
}

typedef struct X { int x; } X;
X fs(_Atomic(X) *c) {
  // CHECK: @fs
  // CHECK: atomicrmw xchg i32*
  return __atomic_exchange(c, (X){2}, memory_order_seq_cst);
}

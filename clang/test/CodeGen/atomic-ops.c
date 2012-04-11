// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

// Also test serialization of atomic operations here, to avoid duplicating the
// test.
// RUN: %clang_cc1 %s -emit-pch -o %t -triple=i686-apple-darwin9
// RUN: %clang_cc1 %s -include-pch %t -triple=i686-apple-darwin9 -emit-llvm -o - | FileCheck %s
#ifndef ALREADY_INCLUDED
#define ALREADY_INCLUDED

// Basic IRGen tests for __c11_atomic_*

// FIXME: Need to implement __c11_atomic_is_lock_free

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

int fi1(_Atomic(int) *i) {
  // CHECK: @fi1
  // CHECK: load atomic i32* {{.*}} seq_cst
  return __c11_atomic_load(i, memory_order_seq_cst);
}

void fi2(_Atomic(int) *i) {
  // CHECK: @fi2
  // CHECK: store atomic i32 {{.*}} seq_cst
  __c11_atomic_store(i, 1, memory_order_seq_cst);
}

void fi3(_Atomic(int) *i) {
  // CHECK: @fi3
  // CHECK: atomicrmw and
  __c11_atomic_fetch_and(i, 1, memory_order_seq_cst);
}

void fi4(_Atomic(int) *i) {
  // CHECK: @fi4
  // CHECK: cmpxchg i32*
  int cmp = 0;
  __c11_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_acquire, memory_order_acquire);
}

float ff1(_Atomic(float) *d) {
  // CHECK: @ff1
  // CHECK: load atomic i32* {{.*}} monotonic
  return __c11_atomic_load(d, memory_order_relaxed);
}

void ff2(_Atomic(float) *d) {
  // CHECK: @ff2
  // CHECK: store atomic i32 {{.*}} release
  __c11_atomic_store(d, 1, memory_order_release);
}

float ff3(_Atomic(float) *d) {
  return __c11_atomic_exchange(d, 2, memory_order_seq_cst);
}

int* fp1(_Atomic(int*) *p) {
  // CHECK: @fp1
  // CHECK: load atomic i32* {{.*}} seq_cst
  return __c11_atomic_load(p, memory_order_seq_cst);
}

int* fp2(_Atomic(int*) *p) {
  // CHECK: @fp2
  // CHECK: store i32 4
  // CHECK: atomicrmw add {{.*}} monotonic
  return __c11_atomic_fetch_add(p, 1, memory_order_relaxed);
}

_Complex float fc(_Atomic(_Complex float) *c) {
  // CHECK: @fc
  // CHECK: atomicrmw xchg i64*
  return __c11_atomic_exchange(c, 2, memory_order_seq_cst);
}

typedef struct X { int x; } X;
X fs(_Atomic(X) *c) {
  // CHECK: @fs
  // CHECK: atomicrmw xchg i32*
  return __c11_atomic_exchange(c, (X){2}, memory_order_seq_cst);
}

int lock_free() {
  // CHECK: @lock_free
  // CHECK: ret i32 1
  return __c11_atomic_is_lock_free(sizeof(_Atomic(int)));
}

// Tests for atomic operations on big values.  These should call the functions
// defined here:
// http://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary#The_Library_interface

struct foo {
  int big[128];
};

_Atomic(struct foo) bigAtomic;

void structAtomicStore() {
  // CHECK: @structAtomicStore
  struct foo f = {0};
  __c11_atomic_store(&bigAtomic, f, 5);
  // CHECK: call void @__atomic_store(i32 512, i8* bitcast (%struct.foo* @bigAtomic to i8*), 
}
void structAtomicLoad() {
  // CHECK: @structAtomicLoad
  struct foo f = __c11_atomic_load(&bigAtomic, 5);
  // CHECK: call void @__atomic_load(i32 512, i8* bitcast (%struct.foo* @bigAtomic to i8*), 
}
struct foo structAtomicExchange() {
  // CHECK: @structAtomicExchange
  struct foo f = {0};
  return __c11_atomic_exchange(&bigAtomic, f, 5);
  // CHECK: call void @__atomic_exchange(i32 512, i8* bitcast (%struct.foo* @bigAtomic to i8*), 
}
int structAtomicCmpExchange() {
  // CHECK: @structAtomicCmpExchange
  struct foo f = {0};
  struct foo g = {0};
  g.big[12] = 12;
  return __c11_atomic_compare_exchange_strong(&bigAtomic, &f, g, 5, 5);
  // CHECK: call zeroext i1 @__atomic_compare_exchange(i32 512, i8* bitcast (%struct.foo* @bigAtomic to i8*), 
}

#endif

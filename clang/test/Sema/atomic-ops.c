// RUN: %clang_cc1 %s -verify -fsyntax-only

// Basic parsing/Sema tests for __c11_atomic_*

// FIXME: Need to implement __c11_atomic_is_lock_free

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

void f(_Atomic(int) *i, _Atomic(int*) *p, _Atomic(float) *d) {
  __c11_atomic_load(0); // expected-error {{too few arguments to function}}
  __c11_atomic_load(0,0,0); // expected-error {{too many arguments to function}}
  __c11_atomic_store(0,0,0); // expected-error {{first argument to atomic operation}}
  __c11_atomic_store((int*)0,0,0); // expected-error {{first argument to atomic operation}}

  __c11_atomic_load(i, memory_order_seq_cst);
  __c11_atomic_load(p, memory_order_seq_cst);
  __c11_atomic_load(d, memory_order_seq_cst);

  __c11_atomic_store(i, 1, memory_order_seq_cst);
  __c11_atomic_store(p, 1, memory_order_seq_cst); // expected-warning {{incompatible integer to pointer conversion}}
  (int)__c11_atomic_store(d, 1, memory_order_seq_cst); // expected-error {{operand of type 'void'}}

  __c11_atomic_fetch_add(i, 1, memory_order_seq_cst);
  __c11_atomic_fetch_add(p, 1, memory_order_seq_cst);
  __c11_atomic_fetch_add(d, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer or pointer}}

  __c11_atomic_fetch_and(i, 1, memory_order_seq_cst);
  __c11_atomic_fetch_and(p, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer}}
  __c11_atomic_fetch_and(d, 1, memory_order_seq_cst); // expected-error {{must be a pointer to atomic integer}}

  __c11_atomic_compare_exchange_strong(i, 0, 1, memory_order_seq_cst, memory_order_seq_cst);
  __c11_atomic_compare_exchange_strong(p, 0, (int*)1, memory_order_seq_cst, memory_order_seq_cst);
  __c11_atomic_compare_exchange_strong(d, (int*)0, 1, memory_order_seq_cst, memory_order_seq_cst); // expected-warning {{incompatible pointer types}}
}

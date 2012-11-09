//===-- tsan_interface_atomic.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_INTERFACE_ATOMIC_H
#define TSAN_INTERFACE_ATOMIC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef char  __tsan_atomic8;
typedef short __tsan_atomic16;  // NOLINT
typedef int   __tsan_atomic32;
typedef long  __tsan_atomic64;  // NOLINT

// Part of ABI, do not change.
// http://llvm.org/viewvc/llvm-project/libcxx/trunk/include/atomic?view=markup
typedef enum {
  __tsan_memory_order_relaxed,
  __tsan_memory_order_consume,
  __tsan_memory_order_acquire,
  __tsan_memory_order_release,
  __tsan_memory_order_acq_rel,
  __tsan_memory_order_seq_cst
} __tsan_memory_order;

__tsan_atomic8 __tsan_atomic8_load(const volatile __tsan_atomic8 *a,
    __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_load(const volatile __tsan_atomic16 *a,
    __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_load(const volatile __tsan_atomic32 *a,
    __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_load(const volatile __tsan_atomic64 *a,
    __tsan_memory_order mo);

void __tsan_atomic8_store(volatile __tsan_atomic8 *a, __tsan_atomic8 v,
    __tsan_memory_order mo);
void __tsan_atomic16_store(volatile __tsan_atomic16 *a, __tsan_atomic16 v,
    __tsan_memory_order mo);
void __tsan_atomic32_store(volatile __tsan_atomic32 *a, __tsan_atomic32 v,
    __tsan_memory_order mo);
void __tsan_atomic64_store(volatile __tsan_atomic64 *a, __tsan_atomic64 v,
    __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_exchange(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_exchange(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_exchange(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_exchange(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_fetch_add(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_fetch_add(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_fetch_add(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_fetch_add(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_fetch_sub(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_fetch_sub(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_fetch_sub(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_fetch_sub(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_fetch_and(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_fetch_and(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_fetch_and(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_fetch_and(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_fetch_or(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_fetch_or(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_fetch_or(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_fetch_or(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_fetch_xor(volatile __tsan_atomic8 *a,
    __tsan_atomic8 v, __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_fetch_xor(volatile __tsan_atomic16 *a,
    __tsan_atomic16 v, __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_fetch_xor(volatile __tsan_atomic32 *a,
    __tsan_atomic32 v, __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_fetch_xor(volatile __tsan_atomic64 *a,
    __tsan_atomic64 v, __tsan_memory_order mo);

int __tsan_atomic8_compare_exchange_weak(volatile __tsan_atomic8 *a,
    __tsan_atomic8 *c, __tsan_atomic8 v, __tsan_memory_order mo);
int __tsan_atomic16_compare_exchange_weak(volatile __tsan_atomic16 *a,
    __tsan_atomic16 *c, __tsan_atomic16 v, __tsan_memory_order mo);
int __tsan_atomic32_compare_exchange_weak(volatile __tsan_atomic32 *a,
    __tsan_atomic32 *c, __tsan_atomic32 v, __tsan_memory_order mo);
int __tsan_atomic64_compare_exchange_weak(volatile __tsan_atomic64 *a,
    __tsan_atomic64 *c, __tsan_atomic64 v, __tsan_memory_order mo);

int __tsan_atomic8_compare_exchange_strong(volatile __tsan_atomic8 *a,
    __tsan_atomic8 *c, __tsan_atomic8 v, __tsan_memory_order mo);
int __tsan_atomic16_compare_exchange_strong(volatile __tsan_atomic16 *a,
    __tsan_atomic16 *c, __tsan_atomic16 v, __tsan_memory_order mo);
int __tsan_atomic32_compare_exchange_strong(volatile __tsan_atomic32 *a,
    __tsan_atomic32 *c, __tsan_atomic32 v, __tsan_memory_order mo);
int __tsan_atomic64_compare_exchange_strong(volatile __tsan_atomic64 *a,
    __tsan_atomic64 *c, __tsan_atomic64 v, __tsan_memory_order mo);

__tsan_atomic8 __tsan_atomic8_compare_exchange_val(
    volatile __tsan_atomic8 *a, __tsan_atomic8 c, __tsan_atomic8 v,
    __tsan_memory_order mo);
__tsan_atomic16 __tsan_atomic16_compare_exchange_val(
    volatile __tsan_atomic16 *a, __tsan_atomic16 c, __tsan_atomic16 v,
    __tsan_memory_order mo);
__tsan_atomic32 __tsan_atomic32_compare_exchange_val(
    volatile __tsan_atomic32 *a, __tsan_atomic32 c, __tsan_atomic32 v,
    __tsan_memory_order mo);
__tsan_atomic64 __tsan_atomic64_compare_exchange_val(
    volatile __tsan_atomic64 *a, __tsan_atomic64 c, __tsan_atomic64 v,
    __tsan_memory_order mo);

void __tsan_atomic_thread_fence(__tsan_memory_order mo);
void __tsan_atomic_signal_fence(__tsan_memory_order mo);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // #ifndef TSAN_INTERFACE_ATOMIC_H

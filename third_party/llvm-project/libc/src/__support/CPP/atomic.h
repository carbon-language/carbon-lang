//===-- A simple equivalent of std::atomic ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_ATOMIC_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_ATOMIC_H

#include "TypeTraits.h"

namespace __llvm_libc {
namespace cpp {

enum class MemoryOrder : int {
  RELAXED = __ATOMIC_RELAXED,
  CONSUME = __ATOMIC_CONSUME,
  ACQUIRE = __ATOMIC_ACQUIRE,
  RELEASE = __ATOMIC_RELEASE,
  ACQ_REL = __ATOMIC_ACQ_REL,
  SEQ_CST = __ATOMIC_SEQ_CST
};

template <typename T> struct Atomic {
  // For now, we will restrict to only arithmetic types.
  static_assert(IsArithmetic<T>::Value, "Only arithmetic types can be atomic.");

private:
  // The value stored should be appropriately aligned so that
  // hardware instructions used to perform atomic operations work
  // correctly.
  static constexpr int ALIGNMENT = sizeof(T) > alignof(T) ? sizeof(T)
                                                          : alignof(T);

public:
  using value_type = T;

  // We keep the internal value public so that it can be addressable.
  // This is useful in places like the Linux futex operations where
  // we need pointers to the memory of the atomic values. Load and store
  // operations should be performed using the atomic methods however.
  alignas(ALIGNMENT) value_type val;

  constexpr Atomic() = default;

  // Intializes the value without using atomic operations.
  constexpr Atomic(value_type v) : val(v) {}

  Atomic(const Atomic &) = delete;
  Atomic &operator=(const Atomic &) = delete;

  // Atomic load
  operator T() { return __atomic_load_n(&val, int(MemoryOrder::SEQ_CST)); }

  T load(MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    return __atomic_load_n(&val, int(mem_ord));
  }

  // Atomic store
  T operator=(T rhs) {
    __atomic_store_n(&val, rhs, int(MemoryOrder::SEQ_CST));
    return rhs;
  }

  void store(T rhs, MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    __atomic_store_n(&val, rhs, int(mem_ord));
  }

  // Atomic compare exchange
  bool compare_exchange_strong(T &expected, T desired,
                               MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    return __atomic_compare_exchange_n(&val, &expected, desired, false,
                                       int(mem_ord), int(mem_ord));
  }

  T exchange(T desired, MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    return __atomic_exchange_n(&val, desired, int(mem_ord));
  }

  T fetch_add(T increment, MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    return __atomic_fetch_add(&val, increment, int(mem_ord));
  }

  T fetch_sub(T decrement, MemoryOrder mem_ord = MemoryOrder::SEQ_CST) {
    return __atomic_fetch_sub(&val, decrement, int(mem_ord));
  }

  // Set the value without using an atomic operation. This is useful
  // in initializing atomic values without a constructor.
  void set(T rhs) { val = rhs; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_ATOMIC_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <atomic>

// template <class T>
// struct atomic
// {
//     bool is_lock_free() const volatile noexcept;
//     bool is_lock_free() const noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     void store(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     T load(memory_order m = memory_order_seq_cst) const volatile noexcept;
//     T load(memory_order m = memory_order_seq_cst) const noexcept;
//     operator T() const volatile noexcept;
//     operator T() const noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) volatile noexcept;
//     T exchange(T desr, memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr, memory_order s, memory_order f) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order s, memory_order f) noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_weak(T& expc, T desr,
//                                memory_order m = memory_order_seq_cst) noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                 memory_order m = memory_order_seq_cst) volatile noexcept;
//     bool compare_exchange_strong(T& expc, T desr,
//                                  memory_order m = memory_order_seq_cst) noexcept;
//
//     atomic() noexcept = default;
//     constexpr atomic(T desr) noexcept;
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     T operator=(T) volatile noexcept;
//     T operator=(T) noexcept;
// };

#include <atomic>
#include <new>
#include <cassert>
#include <thread> // for thread_id
#include <chrono> // for nanoseconds

#include "test_macros.h"

struct TriviallyCopyable {
    TriviallyCopyable ( int i ) : i_(i) {}
    int i_;
    };

template <class T>
void test ( T t ) {
    std::atomic<T> t0(t);
    }

int main(int, char**)
{
    test(TriviallyCopyable(42));
    test(std::this_thread::get_id());
    test(std::chrono::nanoseconds(2));

  return 0;
}

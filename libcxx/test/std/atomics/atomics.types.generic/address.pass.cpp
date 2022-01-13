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
// struct atomic<T*>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(T* desr, memory_order m = memory_order_seq_cst);
//     T* load(memory_order m = memory_order_seq_cst) const volatile;
//     T* load(memory_order m = memory_order_seq_cst) const;
//     operator T*() const volatile;
//     operator T*() const;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst) volatile;
//     T* exchange(T* desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(T*& expc, T* desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(T*& expc, T* desr,
//                                  memory_order m = memory_order_seq_cst);
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_add(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst) volatile;
//     T* fetch_sub(ptrdiff_t op, memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(T* desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//
//     T* operator=(T*) volatile;
//     T* operator=(T*);
//     T* operator++(int) volatile;
//     T* operator++(int);
//     T* operator--(int) volatile;
//     T* operator--(int);
//     T* operator++() volatile;
//     T* operator++();
//     T* operator--() volatile;
//     T* operator--();
//     T* operator+=(ptrdiff_t op) volatile;
//     T* operator+=(ptrdiff_t op);
//     T* operator-=(ptrdiff_t op) volatile;
//     T* operator-=(ptrdiff_t op);
// };

#include <atomic>
#include <new>
#include <type_traits>
#include <cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"

template <class A, class T>
void
do_test()
{
    typedef typename std::remove_pointer<T>::type X;
    A obj(T(0));
    assert(obj == T(0));
    bool b0 = obj.is_lock_free();
    ((void)b0); // mark as unused
    obj.store(T(0));
    assert(obj == T(0));
    obj.store(T(1), std::memory_order_release);
    assert(obj == T(1));
    assert(obj.load() == T(1));
    assert(obj.load(std::memory_order_acquire) == T(1));
    assert(obj.exchange(T(2)) == T(1));
    assert(obj == T(2));
    assert(obj.exchange(T(3), std::memory_order_relaxed) == T(2));
    assert(obj == T(3));
    T x = obj;
    assert(cmpxchg_weak_loop(obj, x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(obj.compare_exchange_weak(x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    x = T(2);
    assert(obj.compare_exchange_strong(x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert((obj = T(0)) == T(0));
    assert(obj == T(0));
    obj = T(2*sizeof(X));
    assert((obj += std::ptrdiff_t(3)) == T(5*sizeof(X)));
    assert(obj == T(5*sizeof(X)));
    assert((obj -= std::ptrdiff_t(3)) == T(2*sizeof(X)));
    assert(obj == T(2*sizeof(X)));

    {
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23};
        A& zero = *new (storage) A();
        assert(zero == T(0));
        zero.~A();
    }
}

template <class A, class T>
void test()
{
    do_test<A, T>();
    do_test<volatile A, T>();
}

int main(int, char**)
{
    test<std::atomic<int*>, int*>();

  return 0;
}

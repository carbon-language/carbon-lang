//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef struct atomic_address
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(void*, memory_order = memory_order_seq_cst) volatile;
//     void store(void*, memory_order = memory_order_seq_cst);
//     void* load(memory_order = memory_order_seq_cst) const volatile;
//     void* load(memory_order = memory_order_seq_cst) const;
//     operator void*() const volatile;
//     operator void*() const;
//     void* exchange(void*, memory_order = memory_order_seq_cst) volatile;
//     void* exchange(void*, memory_order = memory_order_seq_cst);
//     bool compare_exchange_weak(void*&, void*, memory_order,
//                                memory_order) volatile;
//     bool compare_exchange_weak(void*&, void*, memory_order, memory_order);
//     bool compare_exchange_strong(void*&, void*, memory_order,
//                                  memory_order) volatile;
//     bool compare_exchange_strong(void*&, void*, memory_order, memory_order);
//     bool compare_exchange_weak(void*&, void*,
//                                memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(void*&, void*,
//                                memory_order = memory_order_seq_cst);
//     bool compare_exchange_strong(void*&, void*,
//                                  memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(void*&, void*,
//                                  memory_order = memory_order_seq_cst);
//     bool compare_exchange_weak(const void*&, const void*,
//                                memory_order, memory_order) volatile;
//     bool compare_exchange_weak(const void*&, const void*, memory_order,
//                                memory_order);
//     bool compare_exchange_strong(const void*&, const void*, memory_order,
//                                  memory_order) volatile;
//     bool compare_exchange_strong(const void*&, const void*, memory_order,
//                                  memory_order);
//     bool compare_exchange_weak(const void*&, const void*,
//                                memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(const void*&, const void*,
//                                memory_order = memory_order_seq_cst);
//     bool compare_exchange_strong(const void*&, const void*,
//                                  memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(const void*&, const void*,
//                                  memory_order = memory_order_seq_cst);
//     void* fetch_add(ptrdiff_t, memory_order = memory_order_seq_cst) volatile;
//     void* fetch_add(ptrdiff_t, memory_order = memory_order_seq_cst);
//     void* fetch_sub(ptrdiff_t, memory_order = memory_order_seq_cst) volatile;
//     void* fetch_sub(ptrdiff_t, memory_order = memory_order_seq_cst);
//     atomic_address() = default;
//     constexpr atomic_address(void*);
//     atomic_address(const atomic_address&) = delete;
//     atomic_address& operator=(const atomic_address&) = delete;
//     atomic_address& operator=(const atomic_address&) volatile = delete;
//     void* operator=(const void*) volatile;
//     void* operator=(const void*);
//     void* operator+=(ptrdiff_t) volatile;
//     void* operator+=(ptrdiff_t);
//     void* operator-=(ptrdiff_t) volatile;
//     void* operator-=(ptrdiff_t);
// } atomic_address;
// 
// bool atomic_is_lock_free(const volatile atomic_address*);
// bool atomic_is_lock_free(const atomic_address*);
// void atomic_init(volatile atomic_address*, void*);
// void atomic_init(atomic_address*, void*);
// void atomic_store(volatile atomic_address*, void*);
// void atomic_store(atomic_address*, void*);
// void atomic_store_explicit(volatile atomic_address*, void*, memory_order);
// void atomic_store_explicit(atomic_address*, void*, memory_order);
// void* atomic_load(const volatile atomic_address*);
// void* atomic_load(const atomic_address*);
// void* atomic_load_explicit(const volatile atomic_address*, memory_order);
// void* atomic_load_explicit(const atomic_address*, memory_order);
// void* atomic_exchange(volatile atomic_address*, void*);
// void* atomic_exchange(atomic_address*, void*);
// void* atomic_exchange_explicit(volatile atomic_address*, void*, memory_order);
// void* atomic_exchange_explicit(atomic_address*, void*, memory_order);
// bool atomic_compare_exchange_weak(volatile atomic_address*, void**, void*);
// bool atomic_compare_exchange_weak(atomic_address*, void**, void*);
// bool atomic_compare_exchange_strong(volatile atomic_address*, void**, void*);
// bool atomic_compare_exchange_strong(atomic_address*, void**, void*);
// bool atomic_compare_exchange_weak_explicit(volatile atomic_address*, void**,
//                                            void*, memory_order, memory_order);
// bool atomic_compare_exchange_weak_explicit(atomic_address*, void**, void*,
//                                            memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(volatile atomic_address*, void**,
//                                              void*, memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(atomic_address*, void**, void*,
//                                              memory_order, memory_order);
// void* atomic_fetch_add(volatile atomic_address*, ptrdiff_t);
// void* atomic_fetch_add(atomic_address*, ptrdiff_t);
// void* atomic_fetch_add_explicit(volatile atomic_address*, ptrdiff_t,
//                                 memory_order);
// void* atomic_fetch_add_explicit(atomic_address*, ptrdiff_t, memory_order);
// void* atomic_fetch_sub(volatile atomic_address*, ptrdiff_t);
// void* atomic_fetch_sub(atomic_address*, ptrdiff_t);
// void* atomic_fetch_sub_explicit(volatile atomic_address*, ptrdiff_t,
//                                 memory_order);
// void* atomic_fetch_sub_explicit(atomic_address*, ptrdiff_t, memory_order);

#include <atomic>
#include <cassert>

template <class A, class T>
void
do_test()
{
    A obj(T(0));
    assert(obj == T(0));
    std::atomic_init(&obj, T(1));
    assert(obj == T(1));
    std::atomic_init(&obj, T(2));
    assert(obj == T(2));
    bool b0 = obj.is_lock_free();
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
    assert(obj.compare_exchange_weak(x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(obj.compare_exchange_weak(x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert((obj = T(0)) == T(0));
    assert(obj == T(0));
    obj = T(2);
    assert((obj += std::ptrdiff_t(3)) == T(5));
    assert(obj == T(5));
    assert((obj -= std::ptrdiff_t(3)) == T(2));
    assert(obj == T(2));

    std::atomic_init(&obj, T(1));
    assert(obj == T(1));
    bool b1 = std::atomic_is_lock_free(&obj);
    std::atomic_store(&obj, T(0));
    assert(obj == T(0));
    std::atomic_store_explicit(&obj, T(1), std::memory_order_release);
    assert(obj == T(1));
    assert(std::atomic_load(&obj) == T(1));
    assert(std::atomic_load_explicit(&obj, std::memory_order_acquire) == T(1));
    assert(std::atomic_exchange(&obj, T(2)) == T(1));
    assert(obj == T(2));
    assert(std::atomic_exchange_explicit(&obj, T(3), std::memory_order_relaxed) == T(2));
    assert(obj == T(3));
    x = obj;
    assert(std::atomic_compare_exchange_weak(&obj, &x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(std::atomic_compare_exchange_weak(&obj, &x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    assert(std::atomic_compare_exchange_strong(&obj, &x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(std::atomic_compare_exchange_strong(&obj, &x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, T(2),
             std::memory_order_relaxed, std::memory_order_relaxed) == true);
    assert(obj == T(2));
    assert(x == T(1));
    assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, T(3),
            std::memory_order_relaxed, std::memory_order_relaxed) == false);
    assert(obj == T(2));
    assert(x == T(2));
    assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, T(3),
             std::memory_order_relaxed, std::memory_order_relaxed) == true);
    assert(obj == T(3));
    assert(x == T(2));
    assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, T(0),
            std::memory_order_relaxed, std::memory_order_relaxed) == false);
    assert(obj == T(3));
    assert(x == T(3));
    assert((obj = T(1)) == T(1));
    assert(obj == T(1));
    obj = T(2);
    assert(std::atomic_fetch_add(&obj, std::ptrdiff_t(3)) == T(2));
    assert(obj == T(5));
    assert(std::atomic_fetch_add_explicit(&obj, std::ptrdiff_t(3), std::memory_order_seq_cst) == T(5));
    assert(obj == T(8));
    assert(std::atomic_fetch_sub(&obj, std::ptrdiff_t(3)) == T(8));
    assert(obj == T(5));
    assert(std::atomic_fetch_sub_explicit(&obj, std::ptrdiff_t(3), std::memory_order_seq_cst) == T(5));
    assert(obj == T(2));
}

template <class A, class T>
void test()
{
    do_test<A, T>();
    do_test<volatile A, T>();
}

int main()
{
    test<std::atomic_address, void*>();
}

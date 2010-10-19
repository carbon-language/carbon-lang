//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef struct atomic_bool
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(bool, memory_order = memory_order_seq_cst) volatile;
//     void store(bool, memory_order = memory_order_seq_cst);
//     bool load(memory_order = memory_order_seq_cst) const volatile;
//     bool load(memory_order = memory_order_seq_cst) const;
//     operator bool() const volatile;
//     operator bool() const;
//     bool exchange(bool, memory_order = memory_order_seq_cst) volatile;
//     bool exchange(bool, memory_order = memory_order_seq_cst);
//     bool compare_exchange_weak(bool&, bool, memory_order,
//                                memory_order) volatile;
//     bool compare_exchange_weak(bool&, bool, memory_order, memory_order);
//     bool compare_exchange_strong(bool&, bool, memory_order,
//                                  memory_order) volatile;
//     bool compare_exchange_strong(bool&, bool, memory_order, memory_order);
//     bool compare_exchange_weak(bool&, bool,
//                                memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(bool&, bool,
//                                memory_order = memory_order_seq_cst);
//     bool compare_exchange_strong(bool&, bool,
//                                  memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(bool&, bool,
//                                  memory_order = memory_order_seq_cst);
//     atomic_bool() = default;
//     constexpr atomic_bool(bool);
//     atomic_bool(const atomic_bool&) = delete;
//     atomic_bool& operator=(const atomic_bool&) = delete;
//     atomic_bool& operator=(const atomic_bool&) volatile = delete;
//     bool operator=(bool) volatile;
//     bool operator=(bool);
// } atomic_bool;
// 
// bool atomic_is_lock_free(const volatile atomic_bool*);
// bool atomic_is_lock_free(const atomic_bool*);
// void atomic_init(volatile atomic_bool*, bool);
// void atomic_init(atomic_bool*, bool);
// void atomic_store(volatile atomic_bool*, bool);
// void atomic_store(atomic_bool*, bool);
// void atomic_store_explicit(volatile atomic_bool*, bool, memory_order);
// void atomic_store_explicit(atomic_bool*, bool, memory_order);
// bool atomic_load(const volatile atomic_bool*);
// bool atomic_load(const atomic_bool*);
// bool atomic_load_explicit(const volatile atomic_bool*, memory_order);
// bool atomic_load_explicit(const atomic_bool*, memory_order);
// bool atomic_exchange(volatile atomic_bool*, bool);
// bool atomic_exchange(atomic_bool*, bool);
// bool atomic_exchange_explicit(volatile atomic_bool*, bool, memory_order);
// bool atomic_exchange_explicit(atomic_bool*, bool, memory_order);
// bool atomic_compare_exchange_weak(volatile atomic_bool*, bool*, bool);
// bool atomic_compare_exchange_weak(atomic_bool*, bool*, bool);
// bool atomic_compare_exchange_strong(volatile atomic_bool*, bool*, bool);
// bool atomic_compare_exchange_strong(atomic_bool*, bool*, bool);
// bool atomic_compare_exchange_weak_explicit(volatile atomic_bool*, bool*, bool,
//                                            memory_order, memory_order);
// bool atomic_compare_exchange_weak_explicit(atomic_bool*, bool*, bool,
//                                            memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(volatile atomic_bool*, bool*, bool,
//                                              memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(atomic_bool*, bool*, bool,
//                                              memory_order, memory_order);

#include <atomic>
#include <cassert>

int main()
{
    {
        volatile std::atomic_bool obj(true);
        assert(obj == true);
        std::atomic_init(&obj, false);
        assert(obj == false);
        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(obj.compare_exchange_weak(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true) == false);
        assert(obj == false);
        assert(x == false);
        assert(obj.compare_exchange_strong(x, true) == true);
        assert(obj == true);
        assert(x == false);
        assert(obj.compare_exchange_strong(x, false) == false);
        assert(obj == true);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);

        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b1 = std::atomic_is_lock_free(&obj);
        std::atomic_store(&obj, false);
        assert(obj == false);
        std::atomic_store_explicit(&obj, true, std::memory_order_release);
        assert(obj == true);
        assert(std::atomic_load(&obj) == true);
        assert(std::atomic_load_explicit(&obj, std::memory_order_acquire) == true);
        assert(std::atomic_exchange(&obj, false) == true);
        assert(obj == false);
        assert(std::atomic_exchange_explicit(&obj, true, std::memory_order_relaxed) == false);
        assert(obj == true);
        x = obj;
        assert(std::atomic_compare_exchange_weak(&obj, &x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak(&obj, &x, true) == false);
        assert(obj == false);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong(&obj, &x, true) == true);
        assert(obj == true);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong(&obj, &x, false) == false);
        assert(obj == true);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, false,
                 std::memory_order_relaxed, std::memory_order_relaxed) == true);
        assert(obj == false);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, true,
                std::memory_order_relaxed, std::memory_order_relaxed) == false);
        assert(obj == false);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, true,
                 std::memory_order_relaxed, std::memory_order_relaxed) == true);
        assert(obj == true);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, false,
                std::memory_order_relaxed, std::memory_order_relaxed) == false);
        assert(obj == true);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
    }
    {
        std::atomic_bool obj(true);
        assert(obj == true);
        std::atomic_init(&obj, false);
        assert(obj == false);
        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b0 = obj.is_lock_free();
        obj.store(false);
        assert(obj == false);
        obj.store(true, std::memory_order_release);
        assert(obj == true);
        assert(obj.load() == true);
        assert(obj.load(std::memory_order_acquire) == true);
        assert(obj.exchange(false) == true);
        assert(obj == false);
        assert(obj.exchange(true, std::memory_order_relaxed) == false);
        assert(obj == true);
        bool x = obj;
        assert(obj.compare_exchange_weak(x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(obj.compare_exchange_weak(x, true) == false);
        assert(obj == false);
        assert(x == false);
        assert(obj.compare_exchange_strong(x, true) == true);
        assert(obj == true);
        assert(x == false);
        assert(obj.compare_exchange_strong(x, false) == false);
        assert(obj == true);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);

        std::atomic_init(&obj, true);
        assert(obj == true);
        bool b1 = std::atomic_is_lock_free(&obj);
        std::atomic_store(&obj, false);
        assert(obj == false);
        std::atomic_store_explicit(&obj, true, std::memory_order_release);
        assert(obj == true);
        assert(std::atomic_load(&obj) == true);
        assert(std::atomic_load_explicit(&obj, std::memory_order_acquire) == true);
        assert(std::atomic_exchange(&obj, false) == true);
        assert(obj == false);
        assert(std::atomic_exchange_explicit(&obj, true, std::memory_order_relaxed) == false);
        assert(obj == true);
        x = obj;
        assert(std::atomic_compare_exchange_weak(&obj, &x, false) == true);
        assert(obj == false);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak(&obj, &x, true) == false);
        assert(obj == false);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong(&obj, &x, true) == true);
        assert(obj == true);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong(&obj, &x, false) == false);
        assert(obj == true);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, false,
                 std::memory_order_relaxed, std::memory_order_relaxed) == true);
        assert(obj == false);
        assert(x == true);
        assert(std::atomic_compare_exchange_weak_explicit(&obj, &x, true,
                std::memory_order_relaxed, std::memory_order_relaxed) == false);
        assert(obj == false);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, true,
                 std::memory_order_relaxed, std::memory_order_relaxed) == true);
        assert(obj == true);
        assert(x == false);
        assert(std::atomic_compare_exchange_strong_explicit(&obj, &x, false,
                std::memory_order_relaxed, std::memory_order_relaxed) == false);
        assert(obj == true);
        assert(x == true);
        assert((obj = false) == false);
        assert(obj == false);
    }
}

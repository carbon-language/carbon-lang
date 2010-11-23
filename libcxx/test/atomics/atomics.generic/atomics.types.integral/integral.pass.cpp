//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef struct atomic_itype
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(itype, memory_order = memory_order_seq_cst) volatile;
//     void store(itype, memory_order = memory_order_seq_cst);
//     itype load(memory_order = memory_order_seq_cst) const volatile;
//     itype load(memory_order = memory_order_seq_cst) const;
//     operator itype() const volatile;
//     operator itype() const;
//     itype exchange(itype, memory_order = memory_order_seq_cst) volatile;
//     itype exchange(itype, memory_order = memory_order_seq_cst);
//     bool compare_exchange_weak(itype&, itype, memory_order,
//                                memory_order) volatile;
//     bool compare_exchange_weak(itype&, itype, memory_order, memory_order);
//     bool compare_exchange_strong(itype&, itype, memory_order,
//                                  memory_order) volatile;
//     bool compare_exchange_strong(itype&, itype, memory_order, memory_order);
//     bool compare_exchange_weak(itype&, itype,
//                                memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(itype&, itype,
//                                memory_order = memory_order_seq_cst);
//     bool compare_exchange_strong(itype&, itype,
//                                  memory_order = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(itype&, itype,
//                                  memory_order = memory_order_seq_cst);
//     itype fetch_add(itype, memory_order = memory_order_seq_cst) volatile;
//     itype fetch_add(itype, memory_order = memory_order_seq_cst);
//     itype fetch_sub(itype, memory_order = memory_order_seq_cst) volatile;
//     itype fetch_sub(itype, memory_order = memory_order_seq_cst);
//     itype fetch_and(itype, memory_order = memory_order_seq_cst) volatile;
//     itype fetch_and(itype, memory_order = memory_order_seq_cst);
//     itype fetch_or(itype, memory_order = memory_order_seq_cst) volatile;
//     itype fetch_or(itype, memory_order = memory_order_seq_cst);
//     itype fetch_xor(itype, memory_order = memory_order_seq_cst) volatile;
//     itype fetch_xor(itype, memory_order = memory_order_seq_cst);
//     atomic_itype() = default;
//     constexpr atomic_itype(itype);
//     atomic_itype(const atomic_itype&) = delete;
//     atomic_itype& operator=(const atomic_itype&) = delete;
//     atomic_itype& operator=(const atomic_itype&) volatile = delete;
//     itype operator=(itype) volatile;
//     itype operator=(itype);
//     itype operator++(int) volatile;
//     itype operator++(int);
//     itype operator--(int) volatile;
//     itype operator--(int);
//     itype operator++() volatile;
//     itype operator++();
//     itype operator--() volatile;
//     itype operator--();
//     itype operator+=(itype) volatile;
//     itype operator+=(itype);
//     itype operator-=(itype) volatile;
//     itype operator-=(itype);
//     itype operator&=(itype) volatile;
//     itype operator&=(itype);
//     itype operator|=(itype) volatile;
//     itype operator|=(itype);
//     itype operator^=(itype) volatile;
//     itype operator^=(itype);
// } atomic_itype;
// 
// bool atomic_is_lock_free(const volatile atomic_itype*);
// bool atomic_is_lock_free(const atomic_itype*);
// void atomic_init(volatile atomic_itype*, itype);
// void atomic_init(atomic_itype*, itype);
// void atomic_store(volatile atomic_itype*, itype);
// void atomic_store(atomic_itype*, itype);
// void atomic_store_explicit(volatile atomic_itype*, itype, memory_order);
// void atomic_store_explicit(atomic_itype*, itype, memory_order);
// itype atomic_load(const volatile atomic_itype*);
// itype atomic_load(const atomic_itype*);
// itype atomic_load_explicit(const volatile atomic_itype*, memory_order);
// itype atomic_load_explicit(const atomic_itype*, memory_order);
// itype atomic_exchange(volatile atomic_itype*, itype);
// itype atomic_exchange(atomic_itype*, itype);
// itype atomic_exchange_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_exchange_explicit(atomic_itype*, itype, memory_order);
// bool atomic_compare_exchange_weak(volatile atomic_itype*, itype*, itype);
// bool atomic_compare_exchange_weak(atomic_itype*, itype*, itype);
// bool atomic_compare_exchange_strong(volatile atomic_itype*, itype*, itype);
// bool atomic_compare_exchange_strong(atomic_itype*, itype*, itype);
// bool atomic_compare_exchange_weak_explicit(volatile atomic_itype*, itype*, itype,
//                                            memory_order, memory_order);
// bool atomic_compare_exchange_weak_explicit(atomic_itype*, itype*, itype,
//                                            memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(volatile atomic_itype*, itype*, itype,
//                                              memory_order, memory_order);
// bool atomic_compare_exchange_strong_explicit(atomic_itype*, itype*, itype,
//                                              memory_order, memory_order);
// itype atomic_fetch_add(volatile atomic_itype*, itype);
// itype atomic_fetch_add(atomic_itype*, itype);
// itype atomic_fetch_add_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_fetch_add_explicit(atomic_itype*, itype, memory_order);
// itype atomic_fetch_sub(volatile atomic_itype*, itype);
// itype atomic_fetch_sub(atomic_itype*, itype);
// itype atomic_fetch_sub_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_fetch_sub_explicit(atomic_itype*, itype, memory_order);
// itype atomic_fetch_and(volatile atomic_itype*, itype);
// itype atomic_fetch_and(atomic_itype*, itype);
// itype atomic_fetch_and_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_fetch_and_explicit(atomic_itype*, itype, memory_order);
// itype atomic_fetch_or(volatile atomic_itype*, itype);
// itype atomic_fetch_or(atomic_itype*, itype);
// itype atomic_fetch_or_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_fetch_or_explicit(atomic_itype*, itype, memory_order);
// itype atomic_fetch_xor(volatile atomic_itype*, itype);
// itype atomic_fetch_xor(atomic_itype*, itype);
// itype atomic_fetch_xor_explicit(volatile atomic_itype*, itype, memory_order);
// itype atomic_fetch_xor_explicit(atomic_itype*, itype, memory_order);

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
    assert(obj++ == T(0));
    assert(obj == T(1));
    assert(++obj == T(2));
    assert(obj == T(2));
    assert(--obj == T(1));
    assert(obj == T(1));
    assert(obj-- == T(1));
    assert(obj == T(0));
    obj = T(2);
    assert((obj += T(3)) == T(5));
    assert(obj == T(5));
    assert((obj -= T(3)) == T(2));
    assert(obj == T(2));
    assert((obj |= T(5)) == T(7));
    assert(obj == T(7));
    assert((obj &= T(0xF)) == T(7));
    assert(obj == T(7));
    assert((obj ^= T(0xF)) == T(8));
    assert(obj == T(8));

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
    assert(std::atomic_fetch_add(&obj, T(3)) == T(2));
    assert(obj == T(5));
    assert(std::atomic_fetch_add_explicit(&obj, T(3), std::memory_order_seq_cst) == T(5));
    assert(obj == T(8));
    assert(std::atomic_fetch_sub(&obj, T(3)) == T(8));
    assert(obj == T(5));
    assert(std::atomic_fetch_sub_explicit(&obj, T(3), std::memory_order_seq_cst) == T(5));
    assert(obj == T(2));
    assert(std::atomic_fetch_or(&obj, T(5)) == T(2));
    assert(obj == T(7));
    assert(std::atomic_fetch_or_explicit(&obj, T(8), std::memory_order_seq_cst) == T(7));
    assert(obj == T(0xF));
    assert(std::atomic_fetch_and(&obj, T(7)) == T(0xF));
    assert(obj == T(7));
    assert(std::atomic_fetch_and_explicit(&obj, T(3), std::memory_order_seq_cst) == T(7));
    assert(obj == T(3));
    assert(std::atomic_fetch_xor(&obj, T(7)) == T(3));
    assert(obj == T(4));
    assert(std::atomic_fetch_xor_explicit(&obj, T(7), std::memory_order_seq_cst) == T(4));
    assert(obj == T(3));
}

template <class A, class T>
void test()
{
    do_test<A, T>();
    do_test<volatile A, T>();
}


int main()
{
    test<std::atomic_char, char>();
    test<std::atomic_schar, signed char>();
    test<std::atomic_uchar, unsigned char>();
    test<std::atomic_short, short>();
    test<std::atomic_ushort, unsigned short>();
    test<std::atomic_int, int>();
    test<std::atomic_uint, unsigned int>();
    test<std::atomic_long, long>();
    test<std::atomic_ulong, unsigned long>();
    test<std::atomic_llong, long long>();
    test<std::atomic_ullong, unsigned long long>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::atomic_char16_t, char16_t>();
    test<std::atomic_char32_t, char32_t>();
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
    test<std::atomic_wchar_t, wchar_t>();
}

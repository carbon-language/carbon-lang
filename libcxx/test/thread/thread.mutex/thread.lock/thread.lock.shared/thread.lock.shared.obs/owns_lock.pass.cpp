//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <shared_mutex>

// template <class Mutex> class shared_lock;

// bool owns_lock() const noexcept;

#include <shared_mutex>
#include <cassert>

#if _LIBCPP_STD_VER > 11

std::shared_timed_mutex m;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_lock<std::shared_timed_mutex> lk0;
    assert(lk0.owns_lock() == false);
    std::shared_lock<std::shared_timed_mutex> lk1(m);
    assert(lk1.owns_lock() == true);
    lk1.unlock();
    assert(lk1.owns_lock() == false);
    static_assert(noexcept(lk0.owns_lock()), "owns_lock must be noexcept");
#endif  // _LIBCPP_STD_VER > 11
}

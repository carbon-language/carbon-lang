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

// shared_lock& operator=(shared_lock&& u);

#include <shared_mutex>
#include <cassert>

#if _LIBCPP_STD_VER > 11

std::shared_timed_mutex m0;
std::shared_timed_mutex m1;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_lock<std::shared_timed_mutex> lk0(m0);
    std::shared_lock<std::shared_timed_mutex> lk1(m1);
    lk1 = std::move(lk0);
    assert(lk1.mutex() == &m0);
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
#endif  // _LIBCPP_STD_VER > 11
}

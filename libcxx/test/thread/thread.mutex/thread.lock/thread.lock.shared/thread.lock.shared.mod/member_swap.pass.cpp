//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <shared_mutex>

// template <class Mutex> class shared_lock;

// void swap(shared_lock& u) noexcept;

#include <shared_mutex>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct mutex
{
    void lock_shared() {}
    void unlock_shared() {}
};

mutex m;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_lock<mutex> lk1(m);
    std::shared_lock<mutex> lk2;
    lk1.swap(lk2);
    assert(lk1.mutex() == nullptr);
    assert(lk1.owns_lock() == false);
    assert(lk2.mutex() == &m);
    assert(lk2.owns_lock() == true);
    static_assert(noexcept(lk1.swap(lk2)), "member swap must be noexcept");
#endif  // _LIBCPP_STD_VER > 11
}

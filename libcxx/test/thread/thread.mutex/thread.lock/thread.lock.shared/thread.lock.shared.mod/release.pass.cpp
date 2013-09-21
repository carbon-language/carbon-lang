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

// mutex_type* release() noexcept;

#include <shared_mutex>
#include <cassert>

#if _LIBCPP_STD_VER > 11

struct mutex
{
    static int lock_count;
    static int unlock_count;
    void lock_shared() {++lock_count;}
    void unlock_shared() {++unlock_count;}
};

int mutex::lock_count = 0;
int mutex::unlock_count = 0;

mutex m;

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_lock<mutex> lk(m);
    assert(lk.mutex() == &m);
    assert(lk.owns_lock() == true);
    assert(mutex::lock_count == 1);
    assert(mutex::unlock_count == 0);
    assert(lk.release() == &m);
    assert(lk.mutex() == nullptr);
    assert(lk.owns_lock() == false);
    assert(mutex::lock_count == 1);
    assert(mutex::unlock_count == 0);
    static_assert(noexcept(lk.release()), "release must be noexcept");
#endif  // _LIBCPP_STD_VER > 11
}

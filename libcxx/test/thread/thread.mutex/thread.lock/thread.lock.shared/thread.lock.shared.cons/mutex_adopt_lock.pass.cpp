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

// shared_lock(mutex_type& m, adopt_lock_t);

#include <shared_mutex>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    std::shared_timed_mutex m;
    m.lock();
    std::shared_lock<std::shared_timed_mutex> lk(m, std::adopt_lock);
    assert(lk.mutex() == &m);
    assert(lk.owns_lock() == true);
#endif  // _LIBCPP_STD_VER > 11
}

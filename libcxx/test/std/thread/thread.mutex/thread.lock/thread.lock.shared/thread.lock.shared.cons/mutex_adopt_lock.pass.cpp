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
// UNSUPPORTED: c++98, c++03, c++11

// <shared_mutex>

// template <class Mutex> class shared_lock;

// shared_lock(mutex_type& m, adopt_lock_t);

#include <shared_mutex>
#include <cassert>

int main()
{
    std::shared_timed_mutex m;
    m.lock_shared();
    std::shared_lock<std::shared_timed_mutex> lk(m, std::adopt_lock);
    assert(lk.mutex() == &m);
    assert(lk.owns_lock() == true);
}

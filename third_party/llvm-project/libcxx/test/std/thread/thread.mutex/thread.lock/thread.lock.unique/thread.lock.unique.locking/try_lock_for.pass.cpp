//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <mutex>

// template <class Mutex> class unique_lock;

// template <class Rep, class Period>
//   bool try_lock_for(const chrono::duration<Rep, Period>& rel_time);

#include <mutex>
#include <cassert>

#include "test_macros.h"

bool try_lock_for_called = false;

typedef std::chrono::milliseconds ms;

struct mutex
{
    template <class Rep, class Period>
        bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time)
    {
        assert(rel_time == ms(5));
        try_lock_for_called = !try_lock_for_called;
        return try_lock_for_called;
    }
    void unlock() {}
};

mutex m;

int main(int, char**)
{
    std::unique_lock<mutex> lk(m, std::defer_lock);
    assert(lk.try_lock_for(ms(5)) == true);
    assert(try_lock_for_called == true);
    assert(lk.owns_lock() == true);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock_for(ms(5));
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EDEADLK);
    }
#endif
    lk.unlock();
    assert(lk.try_lock_for(ms(5)) == false);
    assert(try_lock_for_called == false);
    assert(lk.owns_lock() == false);
    lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock_for(ms(5));
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif

  return 0;
}

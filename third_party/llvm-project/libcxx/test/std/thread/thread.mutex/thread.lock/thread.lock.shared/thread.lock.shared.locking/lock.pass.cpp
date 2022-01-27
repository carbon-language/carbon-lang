//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11
// ALLOW_RETRIES: 2

// dylib support for shared_mutex was added in macosx10.12
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// <shared_mutex>

// template <class Mutex> class shared_lock;

// void lock();

#include <shared_mutex>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

ms WaitTime = ms(250);

// Thread sanitizer causes more overhead and will sometimes cause this test
// to fail. To prevent this we give Thread sanitizer more time to complete the
// test.
#if !defined(TEST_HAS_SANITIZERS)
ms Tolerance = ms(25);
#else
ms Tolerance = ms(25 * 5);
#endif


void f()
{
    std::shared_lock<std::shared_timed_mutex> lk(m, std::defer_lock);
    time_point t0 = Clock::now();
    lk.lock();
    time_point t1 = Clock::now();
    assert(lk.owns_lock() == true);
    ns d = t1 - t0 - WaitTime;
    assert(d < Tolerance);  // within tolerance
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        lk.lock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EDEADLK);
    }
#endif
    lk.unlock();
    lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        lk.lock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif
}

int main(int, char**)
{
    m.lock();
    std::vector<std::thread> v;
    for (int i = 0; i < 5; ++i)
        v.push_back(support::make_test_thread(f));
    std::this_thread::sleep_for(WaitTime);
    m.unlock();
    for (auto& t : v)
        t.join();

  return 0;
}

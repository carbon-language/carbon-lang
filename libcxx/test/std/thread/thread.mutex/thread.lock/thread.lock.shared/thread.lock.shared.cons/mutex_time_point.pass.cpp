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

// class shared_timed_mutex;

// template <class Clock, class Duration>
//   shared_lock(mutex_type& m, const chrono::time_point<Clock, Duration>& abs_time);

#include <shared_mutex>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cassert>

#if _LIBCPP_STD_VER > 11

std::shared_timed_mutex m;

typedef std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

void f1()
{
    time_point t0 = Clock::now();
    std::shared_lock<std::shared_timed_mutex> lk(m, Clock::now() + ms(300));
    assert(lk.owns_lock() == true);
    time_point t1 = Clock::now();
    ns d = t1 - t0 - ms(250);
    assert(d < ns(50000000));  // within 50ms
}

void f2()
{
    time_point t0 = Clock::now();
    std::shared_lock<std::shared_timed_mutex> lk(m, Clock::now() + ms(250));
    assert(lk.owns_lock() == false);
    time_point t1 = Clock::now();
    ns d = t1 - t0 - ms(250);
    assert(d < ms(50));  // within 50ms
}

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        m.lock();
        std::vector<std::thread> v;
        for (int i = 0; i < 5; ++i)
            v.push_back(std::thread(f1));
        std::this_thread::sleep_for(ms(250));
        m.unlock();
        for (auto& t : v)
            t.join();
    }
    {
        m.lock();
        std::vector<std::thread> v;
        for (int i = 0; i < 5; ++i)
            v.push_back(std::thread(f2));
        std::this_thread::sleep_for(ms(300));
        m.unlock();
        for (auto& t : v)
            t.join();
    }
#endif  // _LIBCPP_STD_VER > 11
}

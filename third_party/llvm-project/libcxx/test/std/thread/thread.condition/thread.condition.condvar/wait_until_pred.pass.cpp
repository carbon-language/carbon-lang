//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ALLOW_RETRIES: 2

// <condition_variable>

// class condition_variable;

// template <class Clock, class Duration, class Predicate>
//     bool
//     wait_until(unique_lock<mutex>& lock,
//                const chrono::time_point<Clock, Duration>& abs_time,
//                Predicate pred);

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

struct Clock
{
    typedef std::chrono::milliseconds duration;
    typedef duration::rep             rep;
    typedef duration::period          period;
    typedef std::chrono::time_point<Clock> time_point;
    static const bool is_steady =  true;

    static time_point now()
    {
        using namespace std::chrono;
        return time_point(duration_cast<duration>(
                steady_clock::now().time_since_epoch()
                                                 ));
    }
};

class Pred
{
    int& i_;
public:
    explicit Pred(int& i) : i_(i) {}

    bool operator()() {return i_ != 0;}
};

std::condition_variable cv;
std::mutex mut;

int test1 = 0;
int test2 = 0;

int runs = 0;

void f()
{
    std::unique_lock<std::mutex> lk(mut);
    assert(test2 == 0);
    test1 = 1;
    cv.notify_one();
    Clock::time_point t0 = Clock::now();
    Clock::time_point t = t0 + Clock::duration(250);
    bool r = cv.wait_until(lk, t, Pred(test2));
    Clock::time_point t1 = Clock::now();
    if (runs == 0)
    {
        assert(t1 - t0 < Clock::duration(250));
        assert(test2 != 0);
        assert(r);
    }
    else
    {
        assert(t1 - t0 - Clock::duration(250) < Clock::duration(50));
        assert(test2 == 0);
        assert(!r);
    }
    ++runs;
}

int main(int, char**)
{
    {
        std::unique_lock<std::mutex> lk(mut);
        std::thread t = support::make_test_thread(f);
        assert(test1 == 0);
        while (test1 == 0)
            cv.wait(lk);
        assert(test1 != 0);
        test2 = 1;
        lk.unlock();
        cv.notify_one();
        t.join();
    }
    test1 = 0;
    test2 = 0;
    {
        std::unique_lock<std::mutex> lk(mut);
        std::thread t = support::make_test_thread(f);
        assert(test1 == 0);
        while (test1 == 0)
            cv.wait(lk);
        assert(test1 != 0);
        lk.unlock();
        t.join();
    }

  return 0;
}

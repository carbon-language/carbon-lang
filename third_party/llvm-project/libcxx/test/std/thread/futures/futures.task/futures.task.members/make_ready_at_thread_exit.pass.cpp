//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// class packaged_task<R(ArgTypes...)>

// void make_ready_at_thread_exit(ArgTypes... args);

#include <future>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

class A
{
    long data_;

public:
    explicit A(long i) : data_(i) {}

    long operator()(long i, long j) const
    {
      if (j == 122)
        TEST_THROW(A(6));
      return data_ + i + j;
    }
};

void func0(std::packaged_task<double(int, char)> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p.make_ready_at_thread_exit(3, 97);
}

void func1(std::packaged_task<double(int, char)> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p.make_ready_at_thread_exit(3, 122);
}

void func2(std::packaged_task<double(int, char)> p)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  p.make_ready_at_thread_exit(3, 97);
  try {
    p.make_ready_at_thread_exit(3, 99);
    }
    catch (const std::future_error& e)
    {
        assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
    }
#else
    ((void)p);
#endif
}

void func3(std::packaged_task<double(int, char)> p)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
      p.make_ready_at_thread_exit(3, 97);
    }
    catch (const std::future_error& e)
    {
        assert(e.code() == make_error_code(std::future_errc::no_state));
    }
#else
    ((void)p);
#endif
}

int main(int, char**)
{
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        support::make_test_thread(func0, std::move(p)).detach();
        assert(f.get() == 105.0);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        support::make_test_thread(func1, std::move(p)).detach();
        try
        {
            f.get();
            assert(false);
        }
        catch (const A& e)
        {
          assert(e(3, 97) == 106.0);
        }
    }
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        support::make_test_thread(func2, std::move(p)).detach();
        assert(f.get() == 105.0);
    }
    {
        std::packaged_task<double(int, char)> p;
        std::thread t = support::make_test_thread(func3, std::move(p));
        t.join();
    }
#endif

  return 0;
}

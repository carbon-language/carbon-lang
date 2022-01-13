// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: fcoroutines-ts
// ADDITIONAL_COMPILE_FLAGS: -fcoroutines-ts

// A simple "breathing" test that checks that <experimental/coroutine>
// can be parsed and used in all dialects, including C++03 in order to match
// Clang's behavior.

#include <experimental/coroutine>

#include "test_macros.h"

namespace coro = std::experimental::coroutines_v1;

coro::suspend_always sa;
coro::suspend_never sn;

struct MyFuture {
  struct promise_type {
    typedef coro::coroutine_handle<promise_type> HandleT;
    coro::suspend_never initial_suspend() { return sn; }
    coro::suspend_always final_suspend() TEST_NOEXCEPT { return sa; }
    coro::suspend_never yield_value(int) { return sn; }
    MyFuture get_return_object() {
      MyFuture f(HandleT::from_promise(*this));
      return f;
    }
    void return_void() {}
    void unhandled_exception() {}
  };
  typedef promise_type::HandleT HandleT;
  MyFuture() : p() {}
  MyFuture(HandleT h) : p(h) {}

  coro::coroutine_handle<promise_type> p;
};

MyFuture test_coro() {
  co_await sn;
  co_yield 42;
  co_return;
}

int main(int, char**)
{
  MyFuture f = test_coro();
  while (!f.p.done())
    f.p.resume();
  f.p.destroy();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <experimental/coroutine>

// template <class Promise>
// struct coroutine_handle<Promise>;

// Promise& promise() const

#include <experimental/coroutine>
#include <type_traits>
#include <memory>
#include <utility>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

struct MyCoro {
  struct promise_type {
    void unhandled_exception() {}
    void return_void() {}
    coro::suspend_never initial_suspend() { return {}; }
    coro::suspend_never final_suspend() noexcept { return {}; }
    MyCoro get_return_object() {
      do_runtime_test();
      return {};
    }
    void do_runtime_test() {
      // Test that a coroutine_handle<const T> can be created from a const
      // promise_type and that it represents the same coroutine as
      // coroutine_handle<T>
      using CH = coro::coroutine_handle<promise_type>;
      using CCH = coro::coroutine_handle<const promise_type>;
      const auto &cthis = *this;
      CH h = CH::from_promise(*this);
      CCH h2 = CCH::from_promise(*this);
      CCH h3 = CCH::from_promise(cthis);
      assert(&h.promise() == this);
      assert(&h2.promise() == this);
      assert(&h3.promise() == this);
      assert(h.address() == h2.address());
      assert(h2.address() == h3.address());
    }
  };
};

MyCoro do_runtime_test() {
  co_await coro::suspend_never{};
}

template <class Promise>
void do_test(coro::coroutine_handle<Promise>&& H) {

  // FIXME Add a runtime test
  {
    ASSERT_SAME_TYPE(decltype(H.promise()), Promise&);
    LIBCPP_ASSERT_NOT_NOEXCEPT(H.promise());
  }
  {
    auto const& CH = H;
    ASSERT_SAME_TYPE(decltype(CH.promise()), Promise&);
    LIBCPP_ASSERT_NOT_NOEXCEPT(CH.promise());
  }
}

int main(int, char**)
{
  do_test(coro::coroutine_handle<int>{});
  do_test(coro::coroutine_handle<const int>{});
  do_runtime_test();

  return 0;
}

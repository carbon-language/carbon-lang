// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <experimental/coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// constexpr coroutine_handle() noexcept
// constexpr coroutine_handle(nullptr_t) noexcept

#include <experimental/coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

template <class C>
void do_test() {
  {
    constexpr C c;
    static_assert(std::is_nothrow_default_constructible<C>::value, "");
    static_assert(c.address() == nullptr, "");
  }
  {
    constexpr C c(nullptr);
    static_assert(std::is_nothrow_constructible<C, std::nullptr_t>::value, "");
    static_assert(c.address() == nullptr, "");
  }
  {
    C c;
    assert(c.address() == nullptr);
  }
  {
    C c(nullptr);
    assert(c.address() == nullptr);
  }
}

int main(int, char**)
{
  do_test<coro::coroutine_handle<>>();
  do_test<coro::coroutine_handle<int>>();

  return 0;
}

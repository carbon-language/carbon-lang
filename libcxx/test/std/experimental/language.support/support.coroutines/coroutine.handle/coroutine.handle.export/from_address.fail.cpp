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

// static coroutine_handle from_address(void*) noexcept

// Test that `from_address` is explicitly ill-formed when called with a typed
// pointer. The user cannot possibly have a typed pointer to the coroutine.
// FIXME: This behavior is an extension, and should upstreamed into the TS or
// the test removed if the TS changes are rejected.

#include <experimental/coroutine>
#include <type_traits>
#include <cassert>

namespace coro = std::experimental;

int main(int, char**)
{
  {
    using H = coro::coroutine_handle<>;
    // expected-error@experimental/coroutine:* 3 {{coroutine_handle<void>::from_address cannot be called with non-void pointers}}
    H::from_address((int*)nullptr); // expected-note {{requested here}}
    H::from_address((const void*)nullptr); // expected-note {{requested here}}
    H::from_address((const char*)nullptr); // expected-note {{requested here}}
  }
  {
    using H = coro::coroutine_handle<int>;
    // expected-error@experimental/coroutine:* 1 {{static_assert failed "coroutine_handle<promise_type>::from_address cannot be used with pointers to the coroutine's promise type; use 'from_promise' instead"}}
    H::from_address((const char*)nullptr); // expected-note {{requested here}}
    // expected-error@experimental/coroutine:* 1 {{coroutine_handle<promise_type>::from_address cannot be called with non-void pointers}}
    H::from_address((int*)nullptr); // expected-note {{requested here}}
  }

  return 0;
}

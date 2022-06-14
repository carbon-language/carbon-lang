//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <future>

// template<class R, class... ArgTypes>
//     class packaged_task<R(ArgTypes...)>
// {
// public:
//     typedef R result_type; // extension

// This is a libc++ extension.

#include <future>
#include <type_traits>

#include "test_macros.h"

struct A {};

int main(int, char**)
{
    static_assert((std::is_same<std::packaged_task<A(int, char)>::result_type, A>::value), "");

  return 0;
}

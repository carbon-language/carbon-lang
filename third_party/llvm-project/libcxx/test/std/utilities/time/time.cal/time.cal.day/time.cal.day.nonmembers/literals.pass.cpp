//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class day;

// constexpr day operator""d(unsigned long long d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    using namespace std::chrono;
    ASSERT_NOEXCEPT(               4d);
    ASSERT_SAME_TYPE(day, decltype(4d));

    static_assert( 7d == day(7), "");
    day d1 = 4d;
    assert (d1 == day(4));
    }

    {
    using namespace std::literals;
    ASSERT_NOEXCEPT(                            4d);
    ASSERT_SAME_TYPE(std::chrono::day, decltype(4d));

    static_assert( 7d == std::chrono::day(7), "");

    std::chrono::day d1 = 4d;
    assert (d1 == std::chrono::day(4));
    }


  return 0;
}

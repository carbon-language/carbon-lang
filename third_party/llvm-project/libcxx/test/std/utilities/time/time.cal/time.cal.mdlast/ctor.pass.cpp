//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month_day_last;

//  constexpr month_day_last(const chrono::month& m) noexcept;
//
//  Effects:  Constructs an object of type month_day_last by initializing m_ with m
//
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month     = std::chrono::month;
    using month_day_last = std::chrono::month_day_last;

    ASSERT_NOEXCEPT(month_day_last{month{1}});

    constexpr month_day_last md0{month{}};
    static_assert( md0.month() == month{}, "");
    static_assert(!md0.ok(),               "");

    constexpr month_day_last md1{std::chrono::January};
    static_assert( md1.month() == std::chrono::January, "");
    static_assert( md1.ok(),                            "");

  return 0;
}

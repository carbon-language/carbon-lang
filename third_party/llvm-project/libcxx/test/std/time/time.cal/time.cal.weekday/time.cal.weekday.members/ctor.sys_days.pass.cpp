//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class weekday;

//  constexpr weekday(const sys_days& dp) noexcept;
//
//  Effects:  Constructs an object of type weekday by computing what day
//              of the week  corresponds to the sys_days dp, and representing
//              that day of the week in wd_
//
//  Remarks: For any value ymd of type year_month_day for which ymd.ok() is true,
//                ymd == year_month_day{sys_days{ymd}} is true.
//
// [Example:
//  If dp represents 1970-01-01, the constructed weekday represents Thursday by storing 4 in wd_.
// —end example]

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using sys_days  = std::chrono::sys_days;
    using days      = std::chrono::days;
    using weekday   = std::chrono::weekday;

    ASSERT_NOEXCEPT(weekday{std::declval<sys_days>()});

    {
    constexpr sys_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr weekday wd{sd};

    static_assert( wd.ok(), "");
    static_assert( wd.c_encoding() == 4, "");
    }

    {
    constexpr sys_days sd{days{10957+32}}; // 2-Feb-2000 was a Wednesday
    constexpr weekday wd{sd};

    static_assert( wd.ok(), "");
    static_assert( wd.c_encoding() == 3, "");
    }


    {
    constexpr sys_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr weekday wd{sd};

    static_assert( wd.ok(), "");
    static_assert( wd.c_encoding() == 2, "");
    }

    {
    sys_days sd{days{-(10957+34)}}; // 29-Nov-1939 was a Wednesday
    weekday wd{sd};

    assert( wd.ok());
    assert( wd.c_encoding() == 3);
    }

    return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_weekday;

//  constexpr year_month_weekday(const sys_days& dp) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday that corresponds
//                to the date represented by dp
//
//  Remarks: For any value ymd of type year_month_weekday for which ymd.ok() is true,
//                ymd == year_month_weekday{sys_days{ymd}} is true.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year               = std::chrono::year;
    using days               = std::chrono::days;
    using sys_days           = std::chrono::sys_days;
    using weekday_indexed    = std::chrono::weekday_indexed;
    using year_month_weekday = std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(year_month_weekday{std::declval<const sys_days>()});

    {
    constexpr sys_days sd{}; // 1-Jan-1970 was a Thursday
    constexpr year_month_weekday ymwd{sd};

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{1970},                                 "");
    static_assert( ymwd.month()           == std::chrono::January,                       "");
    static_assert( ymwd.weekday()         == std::chrono::Thursday,                      "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{std::chrono::Thursday, 1},  "");
    static_assert( ymwd                   == year_month_weekday{sys_days{ymwd}},         ""); // round trip
    }

    {
    constexpr sys_days sd{days{10957+32}}; // 2-Feb-2000 was a Wednesday
    constexpr year_month_weekday ymwd{sd};

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{2000},                                 "");
    static_assert( ymwd.month()           == std::chrono::February,                      "");
    static_assert( ymwd.weekday()         == std::chrono::Wednesday,                     "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{std::chrono::Wednesday, 1}, "");
    static_assert( ymwd                   == year_month_weekday{sys_days{ymwd}},         ""); // round trip
    }


    {
    constexpr sys_days sd{days{-10957}}; // 2-Jan-1940 was a Tuesday
    constexpr year_month_weekday ymwd{sd};

    static_assert( ymwd.ok(),                                                            "");
    static_assert( ymwd.year()            == year{1940},                                 "");
    static_assert( ymwd.month()           == std::chrono::January,                       "");
    static_assert( ymwd.weekday()         == std::chrono::Tuesday,                       "");
    static_assert( ymwd.index()           == 1,                                          "");
    static_assert( ymwd.weekday_indexed() == weekday_indexed{std::chrono::Tuesday, 1},   "");
    static_assert( ymwd                   == year_month_weekday{sys_days{ymwd}},         ""); // round trip
    }

    {
    sys_days sd{days{-(10957+34)}}; // 29-Nov-1939 was a Wednesday
    year_month_weekday ymwd{sd};

    assert( ymwd.ok());
    assert( ymwd.year()            == year{1939});
    assert( ymwd.month()           == std::chrono::November);
    assert( ymwd.weekday()         == std::chrono::Wednesday);
    assert( ymwd.index()           == 5);
    assert((ymwd.weekday_indexed() == weekday_indexed{std::chrono::Wednesday, 5}));
    assert( ymwd                   == year_month_weekday{sys_days{ymwd}}); // round trip
    }

  return 0;
}

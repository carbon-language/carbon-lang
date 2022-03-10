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

// explicit constexpr operator local_days() const noexcept;
//
// Returns: If y_.ok() && m_.ok() && wdi_.weekday().ok(), returns a
//    sys_days that represents the date (index() - 1) * 7 days after the first
//    weekday() of year()/month(). If index() is 0 the returned sys_days
//    represents the date 7 days prior to the first weekday() of
//    year()/month(). Otherwise the returned value is unspecified.
//

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year               = std::chrono::year;
    using month              = std::chrono::month;
    using weekday_indexed    = std::chrono::weekday_indexed;
    using local_days         = std::chrono::local_days;
    using days               = std::chrono::days;
    using year_month_weekday = std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(local_days(std::declval<year_month_weekday>()));

    {
    constexpr year_month_weekday ymwd{year{1970}, month{1}, weekday_indexed{std::chrono::Thursday, 1}};
    constexpr local_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{0}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

    {
    constexpr year_month_weekday ymwd{year{2000}, month{2}, weekday_indexed{std::chrono::Wednesday, 1}};
    constexpr local_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{10957+32}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

    // There's one more leap day between 1/1/40 and 1/1/70
    // when compared to 1/1/70 -> 1/1/2000
    {
    constexpr year_month_weekday ymwd{year{1940}, month{1},weekday_indexed{std::chrono::Tuesday, 1}};
    constexpr local_days sd{ymwd};

    static_assert( sd.time_since_epoch() == days{-10957}, "");
    static_assert( year_month_weekday{sd} == ymwd, ""); // and back
    }

    {
    year_month_weekday ymwd{year{1939}, month{11}, weekday_indexed{std::chrono::Wednesday, 5}};
    local_days sd{ymwd};

    assert( sd.time_since_epoch() == days{-(10957+34)});
    assert( year_month_weekday{sd} == ymwd); // and back
    }

    return 0;
}

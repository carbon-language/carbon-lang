//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day_last;

// constexpr chrono::day day() const noexcept;
//  Returns: wd_

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year                = std::chrono::year;
    using month               = std::chrono::month;
    using month_day_last      = std::chrono::month_day_last;
    using year_month_day_last = std::chrono::year_month_day_last;

    ASSERT_NOEXCEPT(                          std::declval<const year_month_day_last>().month_day_last());
    ASSERT_SAME_TYPE(month_day_last, decltype(std::declval<const year_month_day_last>().month_day_last()));

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month_day_last ymdl(year{1234}, month_day_last{month{i}});
        assert( static_cast<unsigned>(ymdl.month_day_last().month()) == i);
    }

  return 0;
}

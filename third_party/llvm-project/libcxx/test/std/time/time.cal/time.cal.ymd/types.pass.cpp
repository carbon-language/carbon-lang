//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year_month_day = std::chrono::year_month_day;

    static_assert(std::is_trivially_copyable_v<year_month_day>, "");
    static_assert(std::is_standard_layout_v<year_month_day>, "");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// using days = duration<signed integer type of at least 25 bits, ratio_multiply<ratio<24>, hours::period>>;

#include <chrono>
#include <type_traits>
#include <limits>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::chrono::days D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(std::is_signed<Rep>::value, "");
    static_assert(std::is_integral<Rep>::value, "");
    static_assert(std::numeric_limits<Rep>::digits >= 25, "");
    static_assert(std::is_same_v<Period, std::ratio_multiply<std::ratio<24>, std::chrono::hours::period>>, "");

  return 0;
}

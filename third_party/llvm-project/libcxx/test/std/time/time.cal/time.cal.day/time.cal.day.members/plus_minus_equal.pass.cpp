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

// constexpr day& operator+=(const days& d) noexcept;
// constexpr day& operator-=(const days& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename D, typename Ds>
constexpr bool testConstexpr()
{
    D d1{1};
    if (static_cast<unsigned>(d1 += Ds{ 1}) !=  2) return false;
    if (static_cast<unsigned>(d1 += Ds{ 2}) !=  4) return false;
    if (static_cast<unsigned>(d1 += Ds{22}) != 26) return false;
    if (static_cast<unsigned>(d1 -= Ds{ 1}) != 25) return false;
    if (static_cast<unsigned>(d1 -= Ds{ 2}) != 23) return false;
    if (static_cast<unsigned>(d1 -= Ds{22}) !=  1) return false;
    return true;
}

int main(int, char**)
{
    using day  = std::chrono::day;
    using days = std::chrono::days;

    ASSERT_NOEXCEPT(std::declval<day&>() += std::declval<days>());
    ASSERT_NOEXCEPT(std::declval<day&>() -= std::declval<days>());

    ASSERT_SAME_TYPE(day&, decltype(std::declval<day&>() += std::declval<days>()));
    ASSERT_SAME_TYPE(day&, decltype(std::declval<day&>() -= std::declval<days>()));

    static_assert(testConstexpr<day, days>(), "");

    for (unsigned i = 0; i <= 10; ++i)
    {
        day d(i);
        assert(static_cast<unsigned>(d += days{22}) == i + 22);
        assert(static_cast<unsigned>(d)             == i + 22);
        assert(static_cast<unsigned>(d -= days{12}) == i + 10);
        assert(static_cast<unsigned>(d)             == i + 10);
    }

  return 0;
}

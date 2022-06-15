//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month;

//                     month() = default;
//  explicit constexpr month(int m) noexcept;
//  explicit constexpr operator int() const noexcept;

//  Effects: Constructs an object of type month by initializing m_ with m.
//    The value held is unspecified if d is not in the range [0, 255].

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month = std::chrono::month;

    ASSERT_NOEXCEPT(month{});
    ASSERT_NOEXCEPT(month(1));
    ASSERT_NOEXCEPT(static_cast<unsigned>(month(1)));

    constexpr month m0{};
    static_assert(static_cast<unsigned>(m0) == 0, "");

    constexpr month m1{1};
    static_assert(static_cast<unsigned>(m1) == 1, "");

    for (unsigned i = 0; i <= 255; ++i)
    {
        month m(i);
        assert(static_cast<unsigned>(m) == i);
    }

  return 0;
}

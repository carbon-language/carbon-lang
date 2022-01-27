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

// constexpr bool operator==(const month& x, const month& y) noexcept;
// constexpr bool operator!=(const month& x, const month& y) noexcept;
// constexpr bool operator< (const month& x, const month& y) noexcept;
// constexpr bool operator> (const month& x, const month& y) noexcept;
// constexpr bool operator<=(const month& x, const month& y) noexcept;
// constexpr bool operator>=(const month& x, const month& y) noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"


int main(int, char**)
{
    using month = std::chrono::month;

    AssertComparisons6AreNoexcept<month>();
    AssertComparisons6ReturnBool<month>();

    static_assert(testComparisons6Values<month>(0U ,0U), "");
    static_assert(testComparisons6Values<month>(0U, 1U), "");

//  Some 'ok' values as well
    static_assert(testComparisons6Values<month>( 5U,  5U), "");
    static_assert(testComparisons6Values<month>( 5U, 10U), "");

    for (unsigned i = 1; i < 10; ++i)
        for (unsigned j = 10; j < 10; ++j)
            assert(testComparisons6Values<month>(i, j));

  return 0;
}

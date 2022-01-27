//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <chrono>

// floor

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   floor(const duration<Rep, Period>& d);

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class ToDuration, class FromDuration>
void
test(const FromDuration& f, const ToDuration& d)
{
    {
    typedef decltype(std::chrono::floor<ToDuration>(f)) R;
    static_assert((std::is_same<R, ToDuration>::value), "");
    assert(std::chrono::floor<ToDuration>(f) == d);
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(std::chrono::milliseconds( 7290000), std::chrono::hours( 2));
    test(std::chrono::milliseconds(-7290000), std::chrono::hours(-3));
    test(std::chrono::milliseconds( 7290000), std::chrono::minutes( 121));
    test(std::chrono::milliseconds(-7290000), std::chrono::minutes(-122));

    {
//  9000000ms is 2 hours and 30 minutes
    constexpr std::chrono::hours h1 = std::chrono::floor<std::chrono::hours>(std::chrono::milliseconds(9000000));
    static_assert(h1.count() == 2, "");
    constexpr std::chrono::hours h2 = std::chrono::floor<std::chrono::hours>(std::chrono::milliseconds(-9000000));
    static_assert(h2.count() == -3, "");
    }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::min  // noexcept after C++17

#include <chrono>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

int main(int, char**)
{
    assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest());
    assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest());
    assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest());
#if TEST_STD_VER >= 11
    static_assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest(), "");
    static_assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest(), "");
    static_assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest(), "");
#endif

    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<int>::min());
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<double>::min());
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::min());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(std::chrono::duration_values<int>::min());
    ASSERT_NOEXCEPT(std::chrono::duration_values<double>::min());
    ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::min());
#endif

  return 0;
}

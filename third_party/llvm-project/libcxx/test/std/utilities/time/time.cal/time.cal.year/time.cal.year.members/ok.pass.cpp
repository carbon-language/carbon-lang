//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year;

// constexpr bool ok() const noexcept;
//  Returns: min() <= y_ && y_ <= max().
//
//  static constexpr year min() noexcept;
//   Returns year{ 32767};
//  static constexpr year max() noexcept;
//   Returns year{-32767};

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year = std::chrono::year;

    ASSERT_NOEXCEPT(                std::declval<const year>().ok());
    ASSERT_SAME_TYPE(bool, decltype(std::declval<const year>().ok()));

    ASSERT_NOEXCEPT(                year::max());
    ASSERT_SAME_TYPE(year, decltype(year::max()));

    ASSERT_NOEXCEPT(                year::min());
    ASSERT_SAME_TYPE(year, decltype(year::min()));

    static_assert(static_cast<int>(year::min()) == -32767, "");
    static_assert(static_cast<int>(year::max()) ==  32767, "");

    assert(year{-20001}.ok());
    assert(year{ -2000}.ok());
    assert(year{    -1}.ok());
    assert(year{     0}.ok());
    assert(year{     1}.ok());
    assert(year{  2000}.ok());
    assert(year{ 20001}.ok());

    static_assert(!year{-32768}.ok(), "");

  return 0;
}

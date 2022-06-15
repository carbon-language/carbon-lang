//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

// construct double with int

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::chrono::duration<double> d(5);
    assert(d.count() == 5);
#if TEST_STD_VER >= 11
    constexpr std::chrono::duration<double> d2(5);
    static_assert(d2.count() == 5, "");
#endif

  return 0;
}

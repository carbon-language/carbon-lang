//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <chrono>

// ceil

// template <class ToDuration, class Rep, class Period>
//   ToDuration
//   ceil(const duration<Rep, Period>& d);

// ToDuration shall be an instantiation of duration.

#include <chrono>

int main(int, char**)
{
    std::chrono::ceil<int>(std::chrono::milliseconds(3));

  return 0;
}

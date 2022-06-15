//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr inner-iterator::inner-iterator() = default;

#include <ranges>

#include "../types.h"

constexpr bool test() {
  {
    [[maybe_unused]] InnerIterForward i;
  }

  {
    [[maybe_unused]] InnerIterInput i;
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

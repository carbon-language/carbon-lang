//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++03, c++11, c++14

// constexpr byte operator~(byte b) noexcept;

int main(int, char**) {
    constexpr std::byte b1{static_cast<std::byte>(1)};
    constexpr std::byte b2{static_cast<std::byte>(2)};
    constexpr std::byte b8{static_cast<std::byte>(8)};

    static_assert(noexcept(~b1), "" );

    static_assert(std::to_integer<int>(~b1) == 254, "");
    static_assert(std::to_integer<int>(~b2) == 253, "");
    static_assert(std::to_integer<int>(~b8) == 247, "");

  return 0;
}

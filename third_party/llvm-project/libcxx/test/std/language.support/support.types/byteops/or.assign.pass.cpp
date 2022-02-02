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

// constexpr byte& operator |=(byte l, byte r) noexcept;


constexpr std::byte test(std::byte b1, std::byte b2) {
    std::byte bret = b1;
    return bret |= b2;
    }


int main(int, char**) {
    std::byte b;  // not constexpr, just used in noexcept check
    constexpr std::byte b1{static_cast<std::byte>(1)};
    constexpr std::byte b2{static_cast<std::byte>(2)};
    constexpr std::byte b8{static_cast<std::byte>(8)};

    static_assert(noexcept(b |= b), "" );

    static_assert(std::to_integer<int>(test(b1, b2)) ==  3, "");
    static_assert(std::to_integer<int>(test(b1, b8)) ==  9, "");
    static_assert(std::to_integer<int>(test(b2, b8)) == 10, "");

    static_assert(std::to_integer<int>(test(b2, b1)) ==  3, "");
    static_assert(std::to_integer<int>(test(b8, b1)) ==  9, "");
    static_assert(std::to_integer<int>(test(b8, b2)) == 10, "");


  return 0;
}

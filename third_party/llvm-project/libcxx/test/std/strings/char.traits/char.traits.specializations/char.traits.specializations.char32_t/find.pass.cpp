//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char32_t>

// static const char_type* find(const char_type* s, size_t n, const char_type& a);
// constexpr in C++17

#include <string>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 14
constexpr bool test_constexpr()
{
    constexpr const char32_t *p = U"123";
    return std::char_traits<char32_t>::find(p, 3, U'1') == p
        && std::char_traits<char32_t>::find(p, 3, U'2') == p + 1
        && std::char_traits<char32_t>::find(p, 3, U'3') == p + 2
        && std::char_traits<char32_t>::find(p, 3, U'4') == nullptr;
}
#endif

int main(int, char**)
{
    char32_t s1[] = {1, 2, 3};
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(1)) == s1);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(2)) == s1+1);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(3)) == s1+2);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(4)) == 0);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(0)) == 0);
    assert(std::char_traits<char32_t>::find(NULL, 0, char32_t(0)) == 0);

#if TEST_STD_VER > 14
    static_assert(test_constexpr(), "" );
#endif

    return 0;
}

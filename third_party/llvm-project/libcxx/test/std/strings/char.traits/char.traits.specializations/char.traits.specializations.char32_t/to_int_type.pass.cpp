//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char32_t>

// static constexpr int_type to_int_type(char_type c);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
#if TEST_STD_VER >= 11
    assert(std::char_traits<char32_t>::to_int_type(U'a') == U'a');
    assert(std::char_traits<char32_t>::to_int_type(U'A') == U'A');
#endif
    assert(std::char_traits<char32_t>::to_int_type(0) == 0);
#endif // _LIBCPP_HAS_NO_UNICODE_CHARS

  return 0;
}

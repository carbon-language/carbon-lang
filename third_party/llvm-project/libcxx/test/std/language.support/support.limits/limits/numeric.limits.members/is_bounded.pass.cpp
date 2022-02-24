//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// is_bounded

#include <limits>

#include "test_macros.h"

template <class T, bool expected>
void
test()
{
    static_assert(std::numeric_limits<T>::is_bounded == expected, "is_bounded test 1");
    static_assert(std::numeric_limits<const T>::is_bounded == expected, "is_bounded test 2");
    static_assert(std::numeric_limits<volatile T>::is_bounded == expected, "is_bounded test 3");
    static_assert(std::numeric_limits<const volatile T>::is_bounded == expected, "is_bounded test 4");
}

int main(int, char**)
{
    test<bool, true>();
    test<char, true>();
    test<signed char, true>();
    test<unsigned char, true>();
    test<wchar_t, true>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t, true>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t, true>();
    test<char32_t, true>();
#endif
    test<short, true>();
    test<unsigned short, true>();
    test<int, true>();
    test<unsigned int, true>();
    test<long, true>();
    test<unsigned long, true>();
    test<long long, true>();
    test<unsigned long long, true>();
#ifndef _LIBCPP_HAS_NO_INT128
    test<__int128_t, true>();
    test<__uint128_t, true>();
#endif
    test<float, true>();
    test<double, true>();
    test<long double, true>();

  return 0;
}

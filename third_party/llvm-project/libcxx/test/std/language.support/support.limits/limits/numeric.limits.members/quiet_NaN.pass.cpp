//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// quiet_NaN()

#include <limits>
#include <cmath>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test_imp(std::true_type)
{
    assert(std::isnan(std::numeric_limits<T>::quiet_NaN()));
    assert(std::isnan(std::numeric_limits<const T>::quiet_NaN()));
    assert(std::isnan(std::numeric_limits<volatile T>::quiet_NaN()));
    assert(std::isnan(std::numeric_limits<const volatile T>::quiet_NaN()));
}

template <class T>
void
test_imp(std::false_type)
{
    assert(std::numeric_limits<T>::quiet_NaN() == T());
    assert(std::numeric_limits<const T>::quiet_NaN() == T());
    assert(std::numeric_limits<volatile T>::quiet_NaN() == T());
    assert(std::numeric_limits<const volatile T>::quiet_NaN() == T());
}

template <class T>
inline
void
test()
{
    test_imp<T>(std::is_floating_point<T>());
}

int main(int, char**)
{
    test<bool>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<wchar_t>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>();
#endif
    test<char16_t>();
    test<char32_t>();
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}

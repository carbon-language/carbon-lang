//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// make_unsigned

#include <type_traits>

#include "test_macros.h"

enum Enum {zero, one_};

#if TEST_STD_VER >= 11
enum BigEnum : unsigned long long // MSVC's ABI doesn't follow the Standard
#else
enum BigEnum
#endif
{
    bigzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

#if !defined(TEST_HAS_NO_INT128) && TEST_STD_VER >= 11
enum HugeEnum : __int128_t
{
    hugezero
};
#endif

template <class T, class U>
void test_make_unsigned()
{
    ASSERT_SAME_TYPE(U, typename std::make_unsigned<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(U, std::make_unsigned_t<T>);
#endif
}

int main(int, char**)
{
    test_make_unsigned<signed char, unsigned char> ();
    test_make_unsigned<unsigned char, unsigned char> ();
    test_make_unsigned<char, unsigned char> ();
    test_make_unsigned<short, unsigned short> ();
    test_make_unsigned<unsigned short, unsigned short> ();
    test_make_unsigned<int, unsigned int> ();
    test_make_unsigned<unsigned int, unsigned int> ();
    test_make_unsigned<long, unsigned long> ();
    test_make_unsigned<unsigned long, unsigned long> ();
    test_make_unsigned<long long, unsigned long long> ();
    test_make_unsigned<unsigned long long, unsigned long long> ();
    test_make_unsigned<wchar_t, std::conditional<sizeof(wchar_t) == 4, unsigned int, unsigned short>::type> ();
    test_make_unsigned<const wchar_t, std::conditional<sizeof(wchar_t) == 4, const unsigned int, const unsigned short>::type> ();
    test_make_unsigned<const Enum, std::conditional<sizeof(Enum) == sizeof(int), const unsigned int, const unsigned char>::type >();
    test_make_unsigned<BigEnum,
                   std::conditional<sizeof(long) == 4, unsigned long long, unsigned long>::type> ();
#ifndef TEST_HAS_NO_INT128
    test_make_unsigned<__int128_t, __uint128_t>();
    test_make_unsigned<__uint128_t, __uint128_t>();
# if TEST_STD_VER >= 11
    test_make_unsigned<HugeEnum, __uint128_t>();
# endif
#endif

  return 0;
}

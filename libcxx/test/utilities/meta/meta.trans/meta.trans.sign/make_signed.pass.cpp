//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// make_signed

#include <type_traits>

enum Enum {zero, one_};

enum BigEnum
{
    bzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

template <class T, class U>
void test_make_signed()
{
    static_assert((std::is_same<typename std::make_signed<T>::type, U>::value), "");
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::make_signed_t<T>, U>::value), "");
#endif
}

int main()
{
    test_make_signed< signed char, signed char >();
    test_make_signed< unsigned char, signed char >();
    test_make_signed< char, signed char >();
    test_make_signed< short, signed short >();
    test_make_signed< unsigned short, signed short >();
    test_make_signed< int, signed int >();
    test_make_signed< unsigned int, signed int >();
    test_make_signed< long, signed long >();
    test_make_signed< unsigned long, long >();
    test_make_signed< long long, signed long long >();
    test_make_signed< unsigned long long, signed long long >();
    test_make_signed< wchar_t, int >();
    test_make_signed< const wchar_t, const int >();
    test_make_signed< const Enum, const int >();
    test_make_signed< BigEnum, std::conditional<sizeof(long) == 4, long long, long>::type >();
}

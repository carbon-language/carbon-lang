//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// make_unsigned

#include <type_traits>

enum Enum {zero, one_};

enum BigEnum
{
    bzero,
    big = 0xFFFFFFFFFFFFFFFFULL
};

template <class T, class U>
void test_make_unsigned()
{
    static_assert((std::is_same<typename std::make_unsigned<T>::type, U>::value), "");
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::make_unsigned_t<T>, U>::value), "");
#endif
}

int main()
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
    test_make_unsigned<const Enum, const unsigned int> ();
    test_make_unsigned<BigEnum,
                   std::conditional<sizeof(long) == 4, unsigned long long, unsigned long>::type> ();
}

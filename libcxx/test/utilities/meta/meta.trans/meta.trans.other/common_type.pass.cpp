//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// common_type

#include <type_traits>

int main()
{
    static_assert((std::is_same<std::common_type<int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<char>::type, char>::value), "");
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<int>, int>::value), "");
    static_assert((std::is_same<std::common_type_t<char>, char>::value), "");
#endif

    static_assert((std::is_same<std::common_type<double, char>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<short, char>::type, int>::value), "");
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<short, char>, int>::value), "");
#endif

    static_assert((std::is_same<std::common_type<double, char, long long>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<unsigned, char, long long>::type, long long>::value), "");
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char, long long>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<unsigned, char, long long>, long long>::value), "");
#endif
}

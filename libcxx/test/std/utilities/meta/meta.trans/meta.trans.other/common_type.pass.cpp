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

#include "test_macros.h"

struct E {};

template <class T>
struct X { explicit X(T const&){} };

template <class T>
struct S { explicit S(T const&){} };

namespace std
{
    template <typename T>
    struct common_type<T, ::S<T> >
    {
        typedef S<T> type;
    };
}

#if TEST_STD_VER >= 11
template <class Tp>
struct always_bool_imp { using type = bool; };
template <class Tp> using always_bool = typename always_bool_imp<Tp>::type;

template <class ...Args>
constexpr auto no_common_type_imp(int)
  -> always_bool<typename std::common_type<Args...>::type>
  { return false; }

template <class ...Args>
constexpr bool no_common_type_imp(long) { return true; }

template <class ...Args>
using no_common_type = std::integral_constant<bool, no_common_type_imp<Args...>(0)>;

#endif // TEST_STD_VER >= 11

int main()
{
    static_assert((std::is_same<std::common_type<int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<char>::type, char>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<int>,   int>::value), "");
    static_assert((std::is_same<std::common_type_t<char>, char>::value), "");
#endif

    static_assert((std::is_same<std::common_type<               int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<const          int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<      volatile int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<const volatile int>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<int,           int>::type, int>::value), "");
    static_assert((std::is_same<std::common_type<int,     const int>::type, int>::value), "");

    static_assert((std::is_same<std::common_type<long,       const int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<const long,       int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<long,    volatile int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<volatile long,    int>::type, long>::value), "");
    static_assert((std::is_same<std::common_type<const long, const int>::type, long>::value), "");

    static_assert((std::is_same<std::common_type<double, char>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<short, char>::type, int>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<short, char>, int>::value), "");
#endif

    static_assert((std::is_same<std::common_type<double, char, long long>::type, double>::value), "");
    static_assert((std::is_same<std::common_type<unsigned, char, long long>::type, long long>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::common_type_t<double, char, long long>, double>::value), "");
    static_assert((std::is_same<std::common_type_t<unsigned, char, long long>, long long>::value), "");
#endif

    static_assert((std::is_same<std::common_type<               void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const          void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<      volatile void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const volatile void>::type, void>::value), "");

    static_assert((std::is_same<std::common_type<void,       const void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const void,       void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<void,    volatile void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<volatile void,    void>::type, void>::value), "");
    static_assert((std::is_same<std::common_type<const void, const void>::type, void>::value), "");

#if TEST_STD_VER >= 11
    static_assert((no_common_type<void, int>::value), "");
    static_assert((no_common_type<int, void>::value), "");
    static_assert((no_common_type<int, E>::value), "");
    static_assert((no_common_type<int, int, E>::value), "");
    static_assert((no_common_type<int, int, E, int>::value), "");
    static_assert((no_common_type<int, int, int, E>::value), "");
    static_assert((no_common_type<int, X<int> >::value), "");
#endif // TEST_STD_VER >= 11

    static_assert((std::is_same<std::common_type<int, S<int> >::type, S<int> >::value), "");
    static_assert((std::is_same<std::common_type<int, S<int>, S<int> >::type, S<int> >::value), "");
    static_assert((std::is_same<std::common_type<int, int, S<int> >::type, S<int> >::value), "");
}

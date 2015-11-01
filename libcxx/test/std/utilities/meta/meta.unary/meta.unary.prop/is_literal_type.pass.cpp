//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_literal_type

#include <type_traits>

template <class T>
void test_is_literal_type()
{
    static_assert( std::is_literal_type<T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_literal_type_v<T>, "");
//     static_assert( std::is_final_v<T>, "");
//     static_assert( std::is_final_v<const T>, "");
//     static_assert( std::is_final_v<volatile T>, "");
//     static_assert( std::is_final_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_literal_type()
{
    static_assert(!std::is_literal_type<T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_literal_type_v<T>, "");
//     static_assert( std::is_final_v<T>, "");
//     static_assert( std::is_final_v<const T>, "");
//     static_assert( std::is_final_v<volatile T>, "");
//     static_assert( std::is_final_v<const volatile T>, "");
#endif
}

struct A
{
};

struct B
{
    B();
};

int main()
{
    test_is_literal_type<int> ();
    test_is_literal_type<const int> ();
    test_is_literal_type<int&> ();
    test_is_literal_type<volatile int&> ();
    test_is_literal_type<A> ();

    test_is_not_literal_type<B> ();
}

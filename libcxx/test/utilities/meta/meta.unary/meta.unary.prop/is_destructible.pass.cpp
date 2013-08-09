//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_destructible

#include <type_traits>

template <class T>
void test_is_destructible()
{
    static_assert( std::is_destructible<T>::value, "");
    static_assert( std::is_destructible<const T>::value, "");
    static_assert( std::is_destructible<volatile T>::value, "");
    static_assert( std::is_destructible<const volatile T>::value, "");
}

template <class T>
void test_is_not_destructible()
{
    static_assert(!std::is_destructible<T>::value, "");
    static_assert(!std::is_destructible<const T>::value, "");
    static_assert(!std::is_destructible<volatile T>::value, "");
    static_assert(!std::is_destructible<const volatile T>::value, "");
}

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    virtual ~Abstract() = 0;
};

struct A
{
    ~A();
};

int main()
{
    test_is_destructible<A>();
    test_is_destructible<int&>();
    test_is_destructible<Union>();
    test_is_destructible<Empty>();
    test_is_destructible<int>();
    test_is_destructible<double>();
    test_is_destructible<int*>();
    test_is_destructible<const int*>();
    test_is_destructible<char[3]>();
    test_is_destructible<bit_zero>();
    test_is_destructible<int[3]>();

    test_is_not_destructible<int[]>();
    test_is_not_destructible<void>();
    test_is_not_destructible<Abstract>();
#if __has_feature(cxx_access_control_sfinae) 
    test_is_not_destructible<NotEmpty>();
#endif
}

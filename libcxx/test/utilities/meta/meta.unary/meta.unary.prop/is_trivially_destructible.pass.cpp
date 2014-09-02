//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_destructible

#include <type_traits>

template <class T>
void test_is_trivially_destructible()
{
    static_assert( std::is_trivially_destructible<T>::value, "");
    static_assert( std::is_trivially_destructible<const T>::value, "");
    static_assert( std::is_trivially_destructible<volatile T>::value, "");
    static_assert( std::is_trivially_destructible<const volatile T>::value, "");
}

template <class T>
void test_is_not_trivially_destructible()
{
    static_assert(!std::is_trivially_destructible<T>::value, "");
    static_assert(!std::is_trivially_destructible<const T>::value, "");
    static_assert(!std::is_trivially_destructible<volatile T>::value, "");
    static_assert(!std::is_trivially_destructible<const volatile T>::value, "");
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
    virtual void foo() = 0;
};

class AbstractDestructor
{
    virtual ~AbstractDestructor() = 0;
};

struct A
{
    ~A();
};

int main()
{
    test_is_not_trivially_destructible<void>();
    test_is_not_trivially_destructible<A>();
    test_is_not_trivially_destructible<AbstractDestructor>();
    test_is_not_trivially_destructible<NotEmpty>();
    test_is_not_trivially_destructible<char[]>();

    test_is_trivially_destructible<Abstract>();
    test_is_trivially_destructible<int&>();
    test_is_trivially_destructible<Union>();
    test_is_trivially_destructible<Empty>();
    test_is_trivially_destructible<int>();
    test_is_trivially_destructible<double>();
    test_is_trivially_destructible<int*>();
    test_is_trivially_destructible<const int*>();
    test_is_trivially_destructible<char[3]>();
    test_is_trivially_destructible<bit_zero>();
}

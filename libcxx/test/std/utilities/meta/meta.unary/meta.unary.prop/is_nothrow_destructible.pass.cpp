//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_destructible

#include <type_traits>

template <class T>
void test_is_nothrow_destructible()
{
    static_assert( std::is_nothrow_destructible<T>::value, "");
    static_assert( std::is_nothrow_destructible<const T>::value, "");
    static_assert( std::is_nothrow_destructible<volatile T>::value, "");
    static_assert( std::is_nothrow_destructible<const volatile T>::value, "");
}

template <class T>
void test_is_not_nothrow_destructible()
{
    static_assert(!std::is_nothrow_destructible<T>::value, "");
    static_assert(!std::is_nothrow_destructible<const T>::value, "");
    static_assert(!std::is_nothrow_destructible<volatile T>::value, "");
    static_assert(!std::is_nothrow_destructible<const volatile T>::value, "");
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
    test_is_not_nothrow_destructible<void>();
    test_is_not_nothrow_destructible<AbstractDestructor>();
    test_is_not_nothrow_destructible<NotEmpty>();
    test_is_not_nothrow_destructible<char[]>();

#if __has_feature(cxx_noexcept)
    test_is_nothrow_destructible<A>();
#endif
    test_is_nothrow_destructible<int&>();
#if  __has_feature(cxx_unrestricted_unions) 
    test_is_nothrow_destructible<Union>();
#endif
#if __has_feature(cxx_access_control_sfinae)
    test_is_nothrow_destructible<Empty>();
#endif
    test_is_nothrow_destructible<int>();
    test_is_nothrow_destructible<double>();
    test_is_nothrow_destructible<int*>();
    test_is_nothrow_destructible<const int*>();
    test_is_nothrow_destructible<char[3]>();
    test_is_nothrow_destructible<Abstract>();
#if __has_feature(cxx_noexcept)
    test_is_nothrow_destructible<bit_zero>();
#endif
}

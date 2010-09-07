//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_nothrow_copy_constructor

#include <type_traits>

template <class T>
void test_has_nothrow_copy_constructor()
{
    static_assert( std::has_nothrow_copy_constructor<T>::value, "");
    static_assert( std::has_nothrow_copy_constructor<const T>::value, "");
    static_assert( std::has_nothrow_copy_constructor<volatile T>::value, "");
    static_assert( std::has_nothrow_copy_constructor<const volatile T>::value, "");
}

template <class T>
void test_has_not_nothrow_copy_constructor()
{
    static_assert(!std::has_nothrow_copy_constructor<T>::value, "");
    static_assert(!std::has_nothrow_copy_constructor<const T>::value, "");
    static_assert(!std::has_nothrow_copy_constructor<volatile T>::value, "");
    static_assert(!std::has_nothrow_copy_constructor<const volatile T>::value, "");
}

class Empty
{
};

class NotEmpty
{
public:
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
public:
    virtual ~Abstract() = 0;
};

struct A
{
    A(const A&);
};

int main()
{
//    test_has_not_nothrow_copy_constructor<void>();
    test_has_not_nothrow_copy_constructor<A>();
    test_has_not_nothrow_copy_constructor<Abstract>();
//    test_has_not_nothrow_copy_constructor<char[3]>();
//    test_has_not_nothrow_copy_constructor<char[]>();

    test_has_nothrow_copy_constructor<int&>();
    test_has_nothrow_copy_constructor<Union>();
    test_has_nothrow_copy_constructor<Empty>();
    test_has_nothrow_copy_constructor<int>();
    test_has_nothrow_copy_constructor<double>();
    test_has_nothrow_copy_constructor<int*>();
    test_has_nothrow_copy_constructor<const int*>();
//    test_has_nothrow_copy_constructor<NotEmpty>();
    test_has_nothrow_copy_constructor<bit_zero>();
}

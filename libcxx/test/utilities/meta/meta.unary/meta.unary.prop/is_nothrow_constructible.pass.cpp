//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_nothrow_constructible;

#include <type_traits>

template <class T>
void test_is_nothrow_constructible()
{
    static_assert(( std::is_nothrow_constructible<T>::value), "");
}

template <class T, class A0>
void test_is_nothrow_constructible()
{
    static_assert(( std::is_nothrow_constructible<T, A0>::value), "");
}

template <class T>
void test_is_not_nothrow_constructible()
{
    static_assert((!std::is_nothrow_constructible<T>::value), "");
}

template <class T, class A0>
void test_is_not_nothrow_constructible()
{
    static_assert((!std::is_nothrow_constructible<T, A0>::value), "");
}

template <class T, class A0, class A1>
void test_is_not_nothrow_constructible()
{
    static_assert((!std::is_nothrow_constructible<T, A0, A1>::value), "");
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
    A(const A&);
};

int main()
{
    test_is_nothrow_constructible<int> ();
    test_is_nothrow_constructible<int, const int&> ();
    test_is_nothrow_constructible<Empty> ();
    test_is_nothrow_constructible<Empty, const Empty&> ();

    test_is_not_nothrow_constructible<A, int> ();
    test_is_not_nothrow_constructible<A, int, double> ();
    test_is_not_nothrow_constructible<A> ();
}

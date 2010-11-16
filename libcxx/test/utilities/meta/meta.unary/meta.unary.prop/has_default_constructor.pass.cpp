//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_default_constructor

#include <type_traits>

template <class T, bool Result>
void test_has_default_constructor()
{
    static_assert(std::has_default_constructor<T>::value == Result, "");
    static_assert(std::has_default_constructor<const T>::value == Result, "");
    static_assert(std::has_default_constructor<volatile T>::value == Result, "");
    static_assert(std::has_default_constructor<const volatile T>::value == Result, "");
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
    A();
};

int main()
{
    test_has_default_constructor<void, false>();
    test_has_default_constructor<int&, false>();
    test_has_default_constructor<char[], false>();
    test_has_default_constructor<Abstract, false>();

    test_has_default_constructor<A, true>();
    test_has_default_constructor<Union, true>();
    test_has_default_constructor<Empty, true>();
    test_has_default_constructor<int, true>();
    test_has_default_constructor<double, true>();
    test_has_default_constructor<int*, true>();
    test_has_default_constructor<const int*, true>();
    test_has_default_constructor<char[3], true>();
    test_has_default_constructor<NotEmpty, true>();
    test_has_default_constructor<bit_zero, true>();
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_copy_constructor

#include <type_traits>

template <class T, bool Result>
void test_has_copy_constructor()
{
    static_assert(std::has_copy_constructor<T>::value == Result, "");
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
    test_has_copy_constructor<char[3], false>();
    test_has_copy_constructor<char[], false>();
    test_has_copy_constructor<void, false>();
    test_has_copy_constructor<Abstract, false>();

    test_has_copy_constructor<A, true>();
    test_has_copy_constructor<int&, true>();
    test_has_copy_constructor<Union, true>();
    test_has_copy_constructor<Empty, true>();
    test_has_copy_constructor<int, true>();
    test_has_copy_constructor<double, true>();
    test_has_copy_constructor<int*, true>();
    test_has_copy_constructor<const int*, true>();
    test_has_copy_constructor<NotEmpty, true>();
    test_has_copy_constructor<bit_zero, true>();
}

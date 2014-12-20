//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copy_assignable

#include <type_traits>

template <class T>
void test_has_trivially_copy_assignable()
{
    static_assert( std::is_trivially_copy_assignable<T>::value, "");
}

template <class T>
void test_has_not_trivially_copy_assignable()
{
    static_assert(!std::is_trivially_copy_assignable<T>::value, "");
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
    A& operator=(const A&);
};

int main()
{
    test_has_trivially_copy_assignable<int&>();
    test_has_trivially_copy_assignable<Union>();
    test_has_trivially_copy_assignable<Empty>();
    test_has_trivially_copy_assignable<int>();
    test_has_trivially_copy_assignable<double>();
    test_has_trivially_copy_assignable<int*>();
    test_has_trivially_copy_assignable<const int*>();
    test_has_trivially_copy_assignable<bit_zero>();

    test_has_not_trivially_copy_assignable<void>();
    test_has_not_trivially_copy_assignable<A>();
    test_has_not_trivially_copy_assignable<NotEmpty>();
    test_has_not_trivially_copy_assignable<Abstract>();
    test_has_not_trivially_copy_assignable<const Empty>();

}

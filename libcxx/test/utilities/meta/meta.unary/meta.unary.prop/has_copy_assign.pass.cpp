//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_copy_assign

#include <type_traits>

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

struct A
{
    A();
};

int main()
{
    static_assert(( std::has_copy_assign<int>::value), "");
    static_assert((!std::has_copy_assign<const int>::value), "");
    static_assert((!std::has_copy_assign<int[]>::value), "");
    static_assert((!std::has_copy_assign<int[3]>::value), "");
    static_assert((!std::has_copy_assign<int&>::value), "");
    static_assert(( std::has_copy_assign<A>::value), "");
    static_assert(( std::has_copy_assign<bit_zero>::value), "");
    static_assert(( std::has_copy_assign<Union>::value), "");
    static_assert(( std::has_copy_assign<NotEmpty>::value), "");
    static_assert(( std::has_copy_assign<Empty>::value), "");
}

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

#ifndef _LIBCPP_HAS_NO_VARIADICS

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

#endif  // _LIBCPP_HAS_NO_VARIADICS

int main()
{
#ifndef _LIBCPP_HAS_NO_VARIADICS
    static_assert((std::is_nothrow_constructible<int, const int>::value), "");
#endif
}

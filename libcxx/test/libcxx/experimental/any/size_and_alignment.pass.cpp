//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// Check that the size and alignment of any are what we expect.

#include <experimental/any>

int main()
{
    using std::experimental::any;
    static_assert(sizeof(any) == sizeof(void*)*4, "");
    static_assert(alignof(any) == alignof(void*), "");
}

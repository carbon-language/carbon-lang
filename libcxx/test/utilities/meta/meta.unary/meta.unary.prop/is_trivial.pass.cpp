//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivial

#include <type_traits>

int main()
{
    static_assert( std::is_trivial<int>::value, "");
    static_assert(!std::is_trivial<int&>::value, "");
    static_assert(!std::is_trivial<volatile int&>::value, "");
}

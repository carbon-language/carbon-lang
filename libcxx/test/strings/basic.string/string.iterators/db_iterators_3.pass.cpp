//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// Subtract iterators from different containers with <.

#if _LIBCPP_DEBUG2 >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <string>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "../min_allocator.h"

int main()
{
    {
    typedef std::string S;
    S s1;
    S s2;
    int i = s1.begin() - s2.begin();
    assert(false);
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s1;
    S s2;
    int i = s1.begin() - s2.begin();
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif

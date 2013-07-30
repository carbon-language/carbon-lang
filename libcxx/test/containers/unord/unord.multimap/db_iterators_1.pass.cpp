//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Compare iterators from different containers with == or !=.

#if _LIBCPP_DEBUG2 >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <string>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "../../min_allocator.h"

int main()
{
    {
    typedef std::unordered_multimap<int, std::string> C;
    C c1;
    C c2;
    bool b = c1.begin() != c2.begin();
    assert(false);
    }
#if __cplusplus >= 201103L
    {
    typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                        min_allocator<std::pair<const int, std::string>>> C;
    C c1;
    C c2;
    bool b = c1.begin() != c2.begin();
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif

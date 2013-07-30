//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Compare local_iterators from different containers with == or !=.

#if _LIBCPP_DEBUG2 >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <string>
#include <cassert>

int main()
{
    {
    typedef std::unordered_multimap<int, std::string> C;
    C c1;
    c1.insert(std::make_pair(1, "one"));
    C c2;
    c2.insert(std::make_pair(1, "one"));
    C::local_iterator i = c1.begin(c1.bucket(1));
    C::local_iterator j = c2.begin(c2.bucket(1));
    assert(i != j);
    assert(false);
    }
}

#else

int main()
{
}

#endif

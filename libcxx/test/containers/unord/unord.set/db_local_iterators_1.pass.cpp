//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Compare local_iterators from different containers with == or !=.

#if _LIBCPP_DEBUG2 >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_set>
#include <cassert>

int main()
{
    {
    typedef int T;
    typedef std::unordered_set<T> C;
    C c1;
    c1.insert(1);
    C c2;
    c2.insert(1);
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

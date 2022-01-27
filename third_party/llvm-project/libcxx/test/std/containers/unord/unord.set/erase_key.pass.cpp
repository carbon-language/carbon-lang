//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// size_type erase(const key_type& k);

#include <unordered_set>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 11
template <typename Unordered>
bool only_deletions ( const Unordered &whole, const Unordered &part ) {
    typename Unordered::const_iterator w = whole.begin();
    typename Unordered::const_iterator p = part.begin();

    while ( w != whole.end () && p != part.end()) {
        if ( *w == *p )
            p++;
        w++;
        }

    return p == part.end();
}
#endif

int main(int, char**)
{
    {
        typedef std::unordered_set<int> C;
        typedef int P;
        P a[] =
        {
            P(1),
            P(2),
            P(3),
            P(4),
            P(1),
            P(2)
        };
        C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.erase(5) == 0);
        assert(c.size() == 4);
        assert(c.count(1) == 1);
        assert(c.count(2) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(2) == 1);
        assert(c.size() == 3);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(2) == 0);
        assert(c.size() == 3);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(4) == 1);
        assert(c.size() == 2);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);

        assert(c.erase(4) == 0);
        assert(c.size() == 2);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);

        assert(c.erase(1) == 1);
        assert(c.size() == 1);
        assert(c.count(3) == 1);

        assert(c.erase(1) == 0);
        assert(c.size() == 1);
        assert(c.count(3) == 1);

        assert(c.erase(3) == 1);
        assert(c.size() == 0);

        assert(c.erase(3) == 0);
        assert(c.size() == 0);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_set<int, std::hash<int>, std::equal_to<int>, min_allocator<int>> C;
        typedef int P;
        P a[] =
        {
            P(1),
            P(2),
            P(3),
            P(4),
            P(1),
            P(2)
        };
        C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.erase(5) == 0);
        assert(c.size() == 4);
        assert(c.count(1) == 1);
        assert(c.count(2) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(2) == 1);
        assert(c.size() == 3);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(2) == 0);
        assert(c.size() == 3);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);

        assert(c.erase(4) == 1);
        assert(c.size() == 2);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);

        assert(c.erase(4) == 0);
        assert(c.size() == 2);
        assert(c.count(1) == 1);
        assert(c.count(3) == 1);

        assert(c.erase(1) == 1);
        assert(c.size() == 1);
        assert(c.count(3) == 1);

        assert(c.erase(1) == 0);
        assert(c.size() == 1);
        assert(c.count(3) == 1);

        assert(c.erase(3) == 1);
        assert(c.size() == 0);

        assert(c.erase(3) == 0);
        assert(c.size() == 0);
    }
    {
    typedef std::unordered_set<int> C;
    C m, m2;
    for ( int i = 0; i < 10; ++i ) {
        m.insert(i);
        m2.insert(i);
        }

    C::iterator i = m2.begin();
    int ctr = 0;
    while (i != m2.end()) {
        if (ctr++ % 2 == 0)
            m2.erase(i++);
        else
            ++i;
        }

    assert (only_deletions (m, m2));
    }
#endif

  return 0;
}

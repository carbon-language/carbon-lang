//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// iterator insert_after(const_iterator p, const value_type& v);

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), 0);
        assert(i == c.begin());
        assert(c.front() == 0);
        assert(c.front() == 0);
        assert(distance(c.begin(), c.end()) == 1);

        i = c.insert_after(c.cbegin(), 1);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(distance(c.begin(), c.end()) == 2);

        i = c.insert_after(next(c.cbegin()), 2);
        assert(i == next(c.begin(), 2));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(distance(c.begin(), c.end()) == 3);

        i = c.insert_after(c.cbegin(), 3);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin(), 1) == 3);
        assert(*next(c.begin(), 2) == 1);
        assert(*next(c.begin(), 3) == 2);
        assert(distance(c.begin(), c.end()) == 4);
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        typedef C::iterator I;
        C c;
        I i = c.insert_after(c.cbefore_begin(), 0);
        assert(i == c.begin());
        assert(c.front() == 0);
        assert(c.front() == 0);
        assert(distance(c.begin(), c.end()) == 1);

        i = c.insert_after(c.cbegin(), 1);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(distance(c.begin(), c.end()) == 2);

        i = c.insert_after(next(c.cbegin()), 2);
        assert(i == next(c.begin(), 2));
        assert(c.front() == 0);
        assert(*next(c.begin()) == 1);
        assert(*next(c.begin(), 2) == 2);
        assert(distance(c.begin(), c.end()) == 3);

        i = c.insert_after(c.cbegin(), 3);
        assert(i == next(c.begin()));
        assert(c.front() == 0);
        assert(*next(c.begin(), 1) == 3);
        assert(*next(c.begin(), 2) == 1);
        assert(*next(c.begin(), 3) == 2);
        assert(distance(c.begin(), c.end()) == 4);
    }
#endif

  return 0;
}

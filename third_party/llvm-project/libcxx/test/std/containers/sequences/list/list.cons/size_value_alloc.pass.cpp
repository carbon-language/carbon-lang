//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// list(size_type n, const T& value, const Allocator& = Allocator());

#include <list>
#include <cassert>
#include "test_macros.h"
#include "DefaultOnly.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::list<int> l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        std::list<int> l(3, 2, std::allocator<int>());
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        // Add 2 for implementations that dynamically allocate a sentinel node and container proxy.
        std::list<int, limited_allocator<int, 3 + 2> > l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
#if TEST_STD_VER >= 11
    {
        std::list<int, min_allocator<int>> l(3, 2);
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int, min_allocator<int>>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
    {
        std::list<int, min_allocator<int>> l(3, 2, min_allocator<int>());
        assert(l.size() == 3);
        assert(std::distance(l.begin(), l.end()) == 3);
        std::list<int, min_allocator<int>>::const_iterator i = l.begin();
        assert(*i == 2);
        ++i;
        assert(*i == 2);
        ++i;
        assert(*i == 2);
    }
#endif

  return 0;
}

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

// pair<const_iterator, const_iterator> equal_range(const key_type& k) const;

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_set<int> C;
        typedef C::const_iterator I;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c(std::begin(a), std::end(a));
        std::pair<I, I> r = c.equal_range(30);
        assert(std::distance(r.first, r.second) == 1);
        assert(*r.first == 30);
        r = c.equal_range(5);
        assert(std::distance(r.first, r.second) == 0);
        r = c.equal_range(50);
        assert(std::distance(r.first, r.second) == 1);
        assert(*r.first == 50);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_set<int, std::hash<int>, std::equal_to<int>, min_allocator<int>> C;
        typedef C::const_iterator I;
        typedef int P;
        P a[] =
        {
            P(10),
            P(20),
            P(30),
            P(40),
            P(50),
            P(50),
            P(50),
            P(60),
            P(70),
            P(80)
        };
        const C c(std::begin(a), std::end(a));
        std::pair<I, I> r = c.equal_range(30);
        assert(std::distance(r.first, r.second) == 1);
        assert(*r.first == 30);
        r = c.equal_range(5);
        assert(std::distance(r.first, r.second) == 0);
        r = c.equal_range(50);
        assert(std::distance(r.first, r.second) == 1);
        assert(*r.first == 50);
    }
#endif

  return 0;
}

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

// iterator insert(const_iterator p, value_type&& x);

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_set<double> C;
        typedef C::iterator R;
        typedef double P;
        C c;
        typename C::const_iterator e = c.end();
        R r = c.insert(e, P(3.5));
        assert(c.size() == 1);
        assert(*r == 3.5);

        r = c.insert(r, P(3.5));
        assert(c.size() == 1);
        assert(*r == 3.5);

        r = c.insert(c.end(), P(4.5));
        assert(c.size() == 2);
        assert(*r == 4.5);

        r = c.insert(c.end(), P(5.5));
        assert(c.size() == 3);
        assert(*r == 5.5);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_set<MoveOnly> C;
        typedef C::iterator R;
        typedef MoveOnly P;
        C c;
        typename C::const_iterator e = c.end();
        R r = c.insert(e, P(3));
        assert(c.size() == 1);
        assert(*r == 3);

        r = c.insert(r, P(3));
        assert(c.size() == 1);
        assert(*r == 3);

        r = c.insert(c.end(), P(4));
        assert(c.size() == 2);
        assert(*r == 4);

        r = c.insert(c.end(), P(5));
        assert(c.size() == 3);
        assert(*r == 5);
    }
    {
        typedef std::unordered_set<double, std::hash<double>,
                                std::equal_to<double>, min_allocator<double>> C;
        typedef C::iterator R;
        typedef double P;
        C c;
        typename C::const_iterator e = c.end();
        R r = c.insert(e, P(3.5));
        assert(c.size() == 1);
        assert(*r == 3.5);

        r = c.insert(r, P(3.5));
        assert(c.size() == 1);
        assert(*r == 3.5);

        r = c.insert(c.end(), P(4.5));
        assert(c.size() == 2);
        assert(*r == 4.5);

        r = c.insert(c.end(), P(5.5));
        assert(c.size() == 3);
        assert(*r == 5.5);
    }
    {
        typedef std::unordered_set<MoveOnly, std::hash<MoveOnly>,
                            std::equal_to<MoveOnly>, min_allocator<MoveOnly>> C;
        typedef C::iterator R;
        typedef MoveOnly P;
        C c;
        typename C::const_iterator e = c.end();
        R r = c.insert(e, P(3));
        assert(c.size() == 1);
        assert(*r == 3);

        r = c.insert(r, P(3));
        assert(c.size() == 1);
        assert(*r == 3);

        r = c.insert(c.end(), P(4));
        assert(c.size() == 2);
        assert(*r == 4);

        r = c.insert(c.end(), P(5));
        assert(c.size() == 3);
        assert(*r == 5);
    }
#endif // TEST_STD_VER >= 11

  return 0;
}

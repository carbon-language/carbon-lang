//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// void insert(initializer_list<value_type> il);

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_multiset<int> C;
        typedef int P;
        C c;
        c.insert(
                    {
                        P(1),
                        P(2),
                        P(3),
                        P(4),
                        P(1),
                        P(2)
                    }
                );
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
    }
    {
        typedef std::unordered_multiset<int, std::hash<int>,
                                      std::equal_to<int>, min_allocator<int>> C;
        typedef int P;
        C c;
        c.insert(
                    {
                        P(1),
                        P(2),
                        P(3),
                        P(4),
                        P(1),
                        P(2)
                    }
                );
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
    }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <list>

// list(list&& c, const allocator_type& a);

#include <list>
#include <cassert>
#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::list<MoveOnly, test_allocator<MoveOnly> > l(test_allocator<MoveOnly>(5));
        std::list<MoveOnly, test_allocator<MoveOnly> > lo(test_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::list<MoveOnly, test_allocator<MoveOnly> > l2(std::move(l), test_allocator<MoveOnly>(6));
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == test_allocator<MoveOnly>(6));
    }
    {
        std::list<MoveOnly, test_allocator<MoveOnly> > l(test_allocator<MoveOnly>(5));
        std::list<MoveOnly, test_allocator<MoveOnly> > lo(test_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::list<MoveOnly, test_allocator<MoveOnly> > l2(std::move(l), test_allocator<MoveOnly>(5));
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == test_allocator<MoveOnly>(5));
    }
    {
        std::list<MoveOnly, other_allocator<MoveOnly> > l(other_allocator<MoveOnly>(5));
        std::list<MoveOnly, other_allocator<MoveOnly> > lo(other_allocator<MoveOnly>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::list<MoveOnly, other_allocator<MoveOnly> > l2(std::move(l), other_allocator<MoveOnly>(4));
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == other_allocator<MoveOnly>(4));
    }
    {
        std::list<MoveOnly, min_allocator<MoveOnly> > l(min_allocator<MoveOnly>{});
        std::list<MoveOnly, min_allocator<MoveOnly> > lo(min_allocator<MoveOnly>{});
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::list<MoveOnly, min_allocator<MoveOnly> > l2(std::move(l), min_allocator<MoveOnly>());
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == min_allocator<MoveOnly>());
    }

  return 0;
}

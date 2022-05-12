//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// vector(vector&& c, const allocator_type& a);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, test_allocator<bool> > l2(std::move(l), test_allocator<bool>(6));
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == test_allocator<bool>(6));
    }
    {
        std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
        std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, test_allocator<bool> > l2(std::move(l), test_allocator<bool>(5));
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == test_allocator<bool>(5));
    }
    {
        std::vector<bool, other_allocator<bool> > l(other_allocator<bool>(5));
        std::vector<bool, other_allocator<bool> > lo(other_allocator<bool>(5));
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, other_allocator<bool> > l2(std::move(l), other_allocator<bool>(4));
        assert(l2 == lo);
        assert(!l.empty());
        assert(l2.get_allocator() == other_allocator<bool>(4));
    }
    {
        std::vector<bool, min_allocator<bool> > l(min_allocator<bool>{});
        std::vector<bool, min_allocator<bool> > lo(min_allocator<bool>{});
        for (int i = 1; i <= 3; ++i)
        {
            l.push_back(i);
            lo.push_back(i);
        }
        std::vector<bool, min_allocator<bool> > l2(std::move(l), min_allocator<bool>());
        assert(l2 == lo);
        assert(l.empty());
        assert(l2.get_allocator() == min_allocator<bool>());
    }

  return 0;
}

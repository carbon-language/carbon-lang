//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class multimap

// template <class... Args>
//   iterator emplace_hint(const_iterator position, Args&&... args);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::multimap<int, DefaultOnly> M;
        typedef M::iterator R;
        M m;
        assert(DefaultOnly::count == 0);
        R r = m.emplace_hint(m.cend());
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 0);
        assert(m.begin()->second == DefaultOnly());
        assert(DefaultOnly::count == 1);
        r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple());
        assert(r == std::next(m.begin()));
        assert(m.size() == 2);
        assert(std::next(m.begin())->first == 1);
        assert(std::next(m.begin())->second == DefaultOnly());
        assert(DefaultOnly::count == 2);
        r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple());
        assert(r == std::next(m.begin(), 2));
        assert(m.size() == 3);
        assert(std::next(m.begin(), 2)->first == 1);
        assert(std::next(m.begin(), 2)->second == DefaultOnly());
        assert(DefaultOnly::count == 3);
    }
    assert(DefaultOnly::count == 0);
    {
        typedef std::multimap<int, Emplaceable> M;
        typedef M::iterator R;
        M m;
        R r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(2),
                                       std::forward_as_tuple());
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 2);
        assert(m.begin()->second == Emplaceable());
        r = m.emplace_hint(m.cbegin(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple(2, 3.5));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(m.begin()->first == 1);
        assert(m.begin()->second == Emplaceable(2, 3.5));
        r = m.emplace_hint(m.cbegin(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple(3, 3.5));
        assert(r == m.begin());
        assert(m.size() == 3);
        assert(r->first == 1);
        assert(r->second == Emplaceable(3, 3.5));
    }
    {
        typedef std::multimap<int, double> M;
        typedef M::iterator R;
        M m;
        R r = m.emplace_hint(m.cend(), M::value_type(2, 3.5));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 3.5);
    }
    {
        typedef std::multimap<int, DefaultOnly, std::less<int>, min_allocator<std::pair<const int, DefaultOnly>>> M;
        typedef M::iterator R;
        M m;
        assert(DefaultOnly::count == 0);
        R r = m.emplace_hint(m.cend());
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 0);
        assert(m.begin()->second == DefaultOnly());
        assert(DefaultOnly::count == 1);
        r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple());
        assert(r == std::next(m.begin()));
        assert(m.size() == 2);
        assert(std::next(m.begin())->first == 1);
        assert(std::next(m.begin())->second == DefaultOnly());
        assert(DefaultOnly::count == 2);
        r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple());
        assert(r == std::next(m.begin(), 2));
        assert(m.size() == 3);
        assert(std::next(m.begin(), 2)->first == 1);
        assert(std::next(m.begin(), 2)->second == DefaultOnly());
        assert(DefaultOnly::count == 3);
    }
    assert(DefaultOnly::count == 0);
    {
        typedef std::multimap<int, Emplaceable, std::less<int>, min_allocator<std::pair<const int, Emplaceable>>> M;
        typedef M::iterator R;
        M m;
        R r = m.emplace_hint(m.cend(), std::piecewise_construct,
                                       std::forward_as_tuple(2),
                                       std::forward_as_tuple());
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 2);
        assert(m.begin()->second == Emplaceable());
        r = m.emplace_hint(m.cbegin(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple(2, 3.5));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(m.begin()->first == 1);
        assert(m.begin()->second == Emplaceable(2, 3.5));
        r = m.emplace_hint(m.cbegin(), std::piecewise_construct,
                                       std::forward_as_tuple(1),
                                       std::forward_as_tuple(3, 3.5));
        assert(r == m.begin());
        assert(m.size() == 3);
        assert(r->first == 1);
        assert(r->second == Emplaceable(3, 3.5));
    }
    {
        typedef std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
        typedef M::iterator R;
        M m;
        R r = m.emplace_hint(m.cend(), M::value_type(2, 3.5));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 3.5);
    }

  return 0;
}

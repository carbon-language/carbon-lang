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

// void insert(initializer_list<value_type> il);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::multimap<int, double> C;
    typedef C::value_type V;
    C m =
           {
               {1, 1},
               {1, 2},
               {2, 1},
               {2, 2},
               {3, 1},
               {3, 2}
           };
    m.insert(
               {
                   {1, 1.5},
                   {2, 1.5},
                   {3, 1.5},
               }
            );
    assert(m.size() == 9);
    assert(distance(m.begin(), m.end()) == 9);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1, 1));
    assert(*++i == V(1, 2));
    assert(*++i == V(1, 1.5));
    assert(*++i == V(2, 1));
    assert(*++i == V(2, 2));
    assert(*++i == V(2, 1.5));
    assert(*++i == V(3, 1));
    assert(*++i == V(3, 2));
    assert(*++i == V(3, 1.5));
    }
    {
    typedef std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> C;
    typedef C::value_type V;
    C m =
           {
               {1, 1},
               {1, 2},
               {2, 1},
               {2, 2},
               {3, 1},
               {3, 2}
           };
    m.insert(
               {
                   {1, 1.5},
                   {2, 1.5},
                   {3, 1.5},
               }
            );
    assert(m.size() == 9);
    assert(distance(m.begin(), m.end()) == 9);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1, 1));
    assert(*++i == V(1, 2));
    assert(*++i == V(1, 1.5));
    assert(*++i == V(2, 1));
    assert(*++i == V(2, 2));
    assert(*++i == V(2, 1.5));
    assert(*++i == V(3, 1));
    assert(*++i == V(3, 2));
    assert(*++i == V(3, 1.5));
    }

  return 0;
}

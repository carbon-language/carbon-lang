//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// template <class InputIterator>
//     multimap(InputIterator first, InputIterator last);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::pair<const int, double> V;
    V ar[] =
    {
        V(1, 1),
        V(1, 1.5),
        V(1, 2),
        V(2, 1),
        V(2, 1.5),
        V(2, 2),
        V(3, 1),
        V(3, 1.5),
        V(3, 2),
    };
    std::multimap<int, double> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
    assert(m.size() == 9);
    assert(std::distance(m.begin(), m.end()) == 9);
    assert(*m.begin() == V(1, 1));
    assert(*std::next(m.begin()) == V(1, 1.5));
    assert(*std::next(m.begin(), 2) == V(1, 2));
    assert(*std::next(m.begin(), 3) == V(2, 1));
    assert(*std::next(m.begin(), 4) == V(2, 1.5));
    assert(*std::next(m.begin(), 5) == V(2, 2));
    assert(*std::next(m.begin(), 6) == V(3, 1));
    assert(*std::next(m.begin(), 7) == V(3, 1.5));
    assert(*std::next(m.begin(), 8) == V(3, 2));
    }
#if TEST_STD_VER >= 11
    {
    typedef std::pair<const int, double> V;
    V ar[] =
    {
        V(1, 1),
        V(1, 1.5),
        V(1, 2),
        V(2, 1),
        V(2, 1.5),
        V(2, 2),
        V(3, 1),
        V(3, 1.5),
        V(3, 2),
    };
    std::multimap<int, double, std::less<int>, min_allocator<V>> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
    assert(m.size() == 9);
    assert(std::distance(m.begin(), m.end()) == 9);
    assert(*m.begin() == V(1, 1));
    assert(*std::next(m.begin()) == V(1, 1.5));
    assert(*std::next(m.begin(), 2) == V(1, 2));
    assert(*std::next(m.begin(), 3) == V(2, 1));
    assert(*std::next(m.begin(), 4) == V(2, 1.5));
    assert(*std::next(m.begin(), 5) == V(2, 2));
    assert(*std::next(m.begin(), 6) == V(3, 1));
    assert(*std::next(m.begin(), 7) == V(3, 1.5));
    assert(*std::next(m.begin(), 8) == V(3, 2));
    }
#if TEST_STD_VER > 11
    {
    typedef std::pair<const int, double> V;
    V ar[] =
    {
        V(1, 1),
        V(1, 1.5),
        V(1, 2),
        V(2, 1),
        V(2, 1.5),
        V(2, 2),
        V(3, 1),
        V(3, 1.5),
        V(3, 2),
    };
    typedef min_allocator<std::pair<const int, double>> A;
    A a;
    std::multimap<int, double, std::less<int>, A> m(ar, ar+sizeof(ar)/sizeof(ar[0]), a);
    assert(m.size() == 9);
    assert(std::distance(m.begin(), m.end()) == 9);
    assert(*m.begin() == V(1, 1));
    assert(*std::next(m.begin()) == V(1, 1.5));
    assert(*std::next(m.begin(), 2) == V(1, 2));
    assert(*std::next(m.begin(), 3) == V(2, 1));
    assert(*std::next(m.begin(), 4) == V(2, 1.5));
    assert(*std::next(m.begin(), 5) == V(2, 2));
    assert(*std::next(m.begin(), 6) == V(3, 1));
    assert(*std::next(m.begin(), 7) == V(3, 1.5));
    assert(*std::next(m.begin(), 8) == V(3, 2));
    assert(m.get_allocator() == a);
    }
#endif
#endif

  return 0;
}

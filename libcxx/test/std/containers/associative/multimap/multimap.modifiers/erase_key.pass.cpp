//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// size_type erase(const key_type& k);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::multimap<int, double> M;
        typedef std::pair<int, double> P;
        typedef M::size_type I;
        P ar[] =
        {
            P(1, 1),
            P(1, 1.5),
            P(1, 2),
            P(2, 1),
            P(2, 1.5),
            P(2, 2),
            P(3, 1),
            P(3, 1.5),
            P(3, 2),
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 9);
        I i = m.erase(2);
        assert(m.size() == 6);
        assert(i == 3);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);
        assert(next(m.begin(), 3)->first == 3);
        assert(next(m.begin(), 3)->second == 1);
        assert(next(m.begin(), 4)->first == 3);
        assert(next(m.begin(), 4)->second == 1.5);
        assert(next(m.begin(), 5)->first == 3);
        assert(next(m.begin(), 5)->second == 2);

        i = m.erase(2);
        assert(m.size() == 6);
        assert(i == 0);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);
        assert(next(m.begin(), 3)->first == 3);
        assert(next(m.begin(), 3)->second == 1);
        assert(next(m.begin(), 4)->first == 3);
        assert(next(m.begin(), 4)->second == 1.5);
        assert(next(m.begin(), 5)->first == 3);
        assert(next(m.begin(), 5)->second == 2);

        i = m.erase(3);
        assert(i == 3);
        assert(m.size() == 3);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);

        i = m.erase(1);
        assert(m.size() == 0);
        assert(i == 3);
    }
#if TEST_STD_VER >= 11
    {
        typedef std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
        typedef std::pair<int, double> P;
        typedef M::size_type I;
        P ar[] =
        {
            P(1, 1),
            P(1, 1.5),
            P(1, 2),
            P(2, 1),
            P(2, 1.5),
            P(2, 2),
            P(3, 1),
            P(3, 1.5),
            P(3, 2),
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 9);
        I i = m.erase(2);
        assert(m.size() == 6);
        assert(i == 3);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);
        assert(next(m.begin(), 3)->first == 3);
        assert(next(m.begin(), 3)->second == 1);
        assert(next(m.begin(), 4)->first == 3);
        assert(next(m.begin(), 4)->second == 1.5);
        assert(next(m.begin(), 5)->first == 3);
        assert(next(m.begin(), 5)->second == 2);

        i = m.erase(2);
        assert(m.size() == 6);
        assert(i == 0);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);
        assert(next(m.begin(), 3)->first == 3);
        assert(next(m.begin(), 3)->second == 1);
        assert(next(m.begin(), 4)->first == 3);
        assert(next(m.begin(), 4)->second == 1.5);
        assert(next(m.begin(), 5)->first == 3);
        assert(next(m.begin(), 5)->second == 2);

        i = m.erase(3);
        assert(i == 3);
        assert(m.size() == 3);
        assert(next(m.begin(), 0)->first == 1);
        assert(next(m.begin(), 0)->second == 1);
        assert(next(m.begin(), 1)->first == 1);
        assert(next(m.begin(), 1)->second == 1.5);
        assert(next(m.begin(), 2)->first == 1);
        assert(next(m.begin(), 2)->second == 2);

        i = m.erase(1);
        assert(m.size() == 0);
        assert(i == 3);
    }
#endif

  return 0;
}

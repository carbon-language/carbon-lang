//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// iterator erase(const_iterator position);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

struct TemplateConstructor
{
    template<typename T>
    TemplateConstructor (const T&) {}
};

bool operator<(const TemplateConstructor&, const TemplateConstructor&) { return false; }

int main(int, char**)
{
    {
        typedef std::map<int, double> M;
        typedef std::pair<int, double> P;
        typedef M::iterator I;
        P ar[] =
        {
            P(1, 1.5),
            P(2, 2.5),
            P(3, 3.5),
            P(4, 4.5),
            P(5, 5.5),
            P(6, 6.5),
            P(7, 7.5),
            P(8, 8.5),
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 8);
        I i = m.erase(std::next(m.cbegin(), 3));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(m.begin()->first == 1);
        assert(m.begin()->second == 1.5);
        assert(std::next(m.begin())->first == 2);
        assert(std::next(m.begin())->second == 2.5);
        assert(std::next(m.begin(), 2)->first == 3);
        assert(std::next(m.begin(), 2)->second == 3.5);
        assert(std::next(m.begin(), 3)->first == 5);
        assert(std::next(m.begin(), 3)->second == 5.5);
        assert(std::next(m.begin(), 4)->first == 6);
        assert(std::next(m.begin(), 4)->second == 6.5);
        assert(std::next(m.begin(), 5)->first == 7);
        assert(std::next(m.begin(), 5)->second == 7.5);
        assert(std::next(m.begin(), 6)->first == 8);
        assert(std::next(m.begin(), 6)->second == 8.5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 6);
        assert(i == m.begin());
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 3);
        assert(std::next(m.begin())->second == 3.5);
        assert(std::next(m.begin(), 2)->first == 5);
        assert(std::next(m.begin(), 2)->second == 5.5);
        assert(std::next(m.begin(), 3)->first == 6);
        assert(std::next(m.begin(), 3)->second == 6.5);
        assert(std::next(m.begin(), 4)->first == 7);
        assert(std::next(m.begin(), 4)->second == 7.5);
        assert(std::next(m.begin(), 5)->first == 8);
        assert(std::next(m.begin(), 5)->second == 8.5);

        i = m.erase(std::next(m.cbegin(), 5));
        assert(m.size() == 5);
        assert(i == m.end());
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 3);
        assert(std::next(m.begin())->second == 3.5);
        assert(std::next(m.begin(), 2)->first == 5);
        assert(std::next(m.begin(), 2)->second == 5.5);
        assert(std::next(m.begin(), 3)->first == 6);
        assert(std::next(m.begin(), 3)->second == 6.5);
        assert(std::next(m.begin(), 4)->first == 7);
        assert(std::next(m.begin(), 4)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 1));
        assert(m.size() == 4);
        assert(i == std::next(m.begin()));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);
        assert(std::next(m.begin(), 2)->first == 6);
        assert(std::next(m.begin(), 2)->second == 6.5);
        assert(std::next(m.begin(), 3)->first == 7);
        assert(std::next(m.begin(), 3)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 3);
        assert(i == std::next(m.begin(), 2));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);
        assert(std::next(m.begin(), 2)->first == 7);
        assert(std::next(m.begin(), 2)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 2));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 1);
        assert(i == std::next(m.begin(), 0));
        assert(m.begin()->first == 5);
        assert(m.begin()->second == 5.5);

        i = m.erase(m.cbegin());
        assert(m.size() == 0);
        assert(i == m.begin());
        assert(i == m.end());
    }
#if TEST_STD_VER >= 11
    {
        typedef std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
        typedef std::pair<int, double> P;
        typedef M::iterator I;
        P ar[] =
        {
            P(1, 1.5),
            P(2, 2.5),
            P(3, 3.5),
            P(4, 4.5),
            P(5, 5.5),
            P(6, 6.5),
            P(7, 7.5),
            P(8, 8.5),
        };
        M m(ar, ar + sizeof(ar)/sizeof(ar[0]));
        assert(m.size() == 8);
        I i = m.erase(std::next(m.cbegin(), 3));
        assert(m.size() == 7);
        assert(i == std::next(m.begin(), 3));
        assert(m.begin()->first == 1);
        assert(m.begin()->second == 1.5);
        assert(std::next(m.begin())->first == 2);
        assert(std::next(m.begin())->second == 2.5);
        assert(std::next(m.begin(), 2)->first == 3);
        assert(std::next(m.begin(), 2)->second == 3.5);
        assert(std::next(m.begin(), 3)->first == 5);
        assert(std::next(m.begin(), 3)->second == 5.5);
        assert(std::next(m.begin(), 4)->first == 6);
        assert(std::next(m.begin(), 4)->second == 6.5);
        assert(std::next(m.begin(), 5)->first == 7);
        assert(std::next(m.begin(), 5)->second == 7.5);
        assert(std::next(m.begin(), 6)->first == 8);
        assert(std::next(m.begin(), 6)->second == 8.5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 6);
        assert(i == m.begin());
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 3);
        assert(std::next(m.begin())->second == 3.5);
        assert(std::next(m.begin(), 2)->first == 5);
        assert(std::next(m.begin(), 2)->second == 5.5);
        assert(std::next(m.begin(), 3)->first == 6);
        assert(std::next(m.begin(), 3)->second == 6.5);
        assert(std::next(m.begin(), 4)->first == 7);
        assert(std::next(m.begin(), 4)->second == 7.5);
        assert(std::next(m.begin(), 5)->first == 8);
        assert(std::next(m.begin(), 5)->second == 8.5);

        i = m.erase(std::next(m.cbegin(), 5));
        assert(m.size() == 5);
        assert(i == m.end());
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 3);
        assert(std::next(m.begin())->second == 3.5);
        assert(std::next(m.begin(), 2)->first == 5);
        assert(std::next(m.begin(), 2)->second == 5.5);
        assert(std::next(m.begin(), 3)->first == 6);
        assert(std::next(m.begin(), 3)->second == 6.5);
        assert(std::next(m.begin(), 4)->first == 7);
        assert(std::next(m.begin(), 4)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 1));
        assert(m.size() == 4);
        assert(i == std::next(m.begin()));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);
        assert(std::next(m.begin(), 2)->first == 6);
        assert(std::next(m.begin(), 2)->second == 6.5);
        assert(std::next(m.begin(), 3)->first == 7);
        assert(std::next(m.begin(), 3)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 3);
        assert(i == std::next(m.begin(), 2));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);
        assert(std::next(m.begin(), 2)->first == 7);
        assert(std::next(m.begin(), 2)->second == 7.5);

        i = m.erase(std::next(m.cbegin(), 2));
        assert(m.size() == 2);
        assert(i == std::next(m.begin(), 2));
        assert(m.begin()->first == 2);
        assert(m.begin()->second == 2.5);
        assert(std::next(m.begin())->first == 5);
        assert(std::next(m.begin())->second == 5.5);

        i = m.erase(std::next(m.cbegin(), 0));
        assert(m.size() == 1);
        assert(i == std::next(m.begin(), 0));
        assert(m.begin()->first == 5);
        assert(m.begin()->second == 5.5);

        i = m.erase(m.cbegin());
        assert(m.size() == 0);
        assert(i == m.begin());
        assert(i == m.end());
    }
#endif
#if TEST_STD_VER >= 14
    {
    //  This is LWG #2059
        typedef TemplateConstructor T;
        typedef std::map<T, int> C;
        typedef C::iterator I;

        C c;
        T a{0};
        I it = c.find(a);
        if (it != c.end())
            c.erase(it);
    }
#endif

  return 0;
}

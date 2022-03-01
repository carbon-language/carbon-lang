//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

//       iterator begin();
// const_iterator begin() const;
//       iterator end();
// const_iterator end()   const;
//
//       reverse_iterator rbegin();
// const_reverse_iterator rbegin() const;
//       reverse_iterator rend();
// const_reverse_iterator rend()   const;
//
// const_iterator         cbegin()  const;
// const_iterator         cend()    const;
// const_reverse_iterator crbegin() const;
// const_reverse_iterator crend()   const;

#include <map>
#include <cassert>
#include <cstddef>

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
            V(4, 1),
            V(4, 1.5),
            V(4, 2),
            V(5, 1),
            V(5, 1.5),
            V(5, 2),
            V(6, 1),
            V(6, 1.5),
            V(6, 2),
            V(7, 1),
            V(7, 1.5),
            V(7, 2),
            V(8, 1),
            V(8, 1.5),
            V(8, 2)
        };
        std::map<int, double> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(static_cast<std::size_t>(std::distance(m.begin(), m.end())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.rbegin(), m.rend())) == m.size());
        std::map<int, double>::iterator i;
        i = m.begin();
        std::map<int, double>::const_iterator k = i;
        assert(i == k);
        for (int j = 1; static_cast<std::size_t>(j) <= m.size(); ++j, ++i)
        {
            assert(i->first == j);
            assert(i->second == 1);
            i->second = 2.5;
            assert(i->second == 2.5);
        }
        assert(i == m.end());
        for (int j = m.size(); j >= 1; --j)
        {
            --i;
            assert(i->first == j);
            assert(i->second == 2.5);
            i->second = 1;
            assert(i->second == 1);
        }
        assert(i == m.begin());
    }
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
            V(4, 1),
            V(4, 1.5),
            V(4, 2),
            V(5, 1),
            V(5, 1.5),
            V(5, 2),
            V(6, 1),
            V(6, 1.5),
            V(6, 2),
            V(7, 1),
            V(7, 1.5),
            V(7, 2),
            V(8, 1),
            V(8, 1.5),
            V(8, 2)
        };
        const std::map<int, double> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(static_cast<std::size_t>(std::distance(m.begin(), m.end())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.cbegin(), m.cend())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.rbegin(), m.rend())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.crbegin(), m.crend())) == m.size());
        std::map<int, double>::const_iterator i;
        i = m.begin();
        for (int j = 1; static_cast<std::size_t>(j) <= m.size(); ++j, ++i)
        {
            assert(i->first == j);
            assert(i->second == 1);
        }
        assert(i == m.end());
        for (int j = m.size(); j >= 1; --j)
        {
            --i;
            assert(i->first == j);
            assert(i->second == 1);
        }
        assert(i == m.begin());
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
            V(4, 1),
            V(4, 1.5),
            V(4, 2),
            V(5, 1),
            V(5, 1.5),
            V(5, 2),
            V(6, 1),
            V(6, 1.5),
            V(6, 2),
            V(7, 1),
            V(7, 1.5),
            V(7, 2),
            V(8, 1),
            V(8, 1.5),
            V(8, 2)
        };
        std::map<int, double, std::less<int>, min_allocator<V>> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(static_cast<std::size_t>(std::distance(m.begin(), m.end())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.rbegin(), m.rend())) == m.size());
        std::map<int, double, std::less<int>, min_allocator<V>>::iterator i;
        i = m.begin();
        std::map<int, double, std::less<int>, min_allocator<V>>::const_iterator k = i;
        assert(i == k);
        for (int j = 1; static_cast<std::size_t>(j) <= m.size(); ++j, ++i)
        {
            assert(i->first == j);
            assert(i->second == 1);
            i->second = 2.5;
            assert(i->second == 2.5);
        }
        assert(i == m.end());
        for (int j = m.size(); j >= 1; --j)
        {
            --i;
            assert(i->first == j);
            assert(i->second == 2.5);
            i->second = 1;
            assert(i->second == 1);
        }
        assert(i == m.begin());
    }
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
            V(4, 1),
            V(4, 1.5),
            V(4, 2),
            V(5, 1),
            V(5, 1.5),
            V(5, 2),
            V(6, 1),
            V(6, 1.5),
            V(6, 2),
            V(7, 1),
            V(7, 1.5),
            V(7, 2),
            V(8, 1),
            V(8, 1.5),
            V(8, 2)
        };
        const std::map<int, double, std::less<int>, min_allocator<V>> m(ar, ar+sizeof(ar)/sizeof(ar[0]));
        assert(static_cast<std::size_t>(std::distance(m.begin(), m.end())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.cbegin(), m.cend())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.rbegin(), m.rend())) == m.size());
        assert(static_cast<std::size_t>(std::distance(m.crbegin(), m.crend())) == m.size());
        std::map<int, double, std::less<int>, min_allocator<V>>::const_iterator i;
        i = m.begin();
        for (int j = 1; static_cast<std::size_t>(j) <= m.size(); ++j, ++i)
        {
            assert(i->first == j);
            assert(i->second == 1);
        }
        assert(i == m.end());
        for (int j = m.size(); j >= 1; --j)
        {
            --i;
            assert(i->first == j);
            assert(i->second == 1);
        }
        assert(i == m.begin());
    }
#endif
#if TEST_STD_VER > 11
    { // N3644 testing
        typedef std::map<int, double> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
    }
#endif

  return 0;
}

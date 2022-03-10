//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Test nested types and default template args:

// template <class T, class Allocator = allocator<T> >
// class deque;

// iterator, const_iterator

#include <deque>
#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::deque<int> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::deque<int, min_allocator<int>> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
#endif
#if TEST_STD_VER > 11
    { // N3644 testing
        std::deque<int>::iterator ii1{}, ii2{};
        std::deque<int>::iterator ii4 = ii1;
        std::deque<int>::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
        assert (!(ii1 <  cii ));
        assert (!(cii <  ii1 ));
        assert ( (ii1 <= cii ));
        assert ( (cii <= ii1 ));
        assert (!(ii1 >  cii ));
        assert (!(cii >  ii1 ));
        assert ( (ii1 >= cii ));
        assert ( (cii >= ii1 ));
        assert (cii - ii1 == 0);
        assert (ii1 - cii == 0);

//         std::deque<int> c;
//         assert ( ii1 != c.cbegin());
//         assert ( cii != c.begin());
//         assert ( cii != c.cend());
//         assert ( ii1 != c.end());
    }
#endif

  return 0;
}

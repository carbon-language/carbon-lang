//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <class Iterator> reverse_iterator<Iterator>
//     make_reverse_iterator(Iterator i);

#include <iterator>
#include <cassert>

#include "test_iterators.h"

#if _LIBCPP_STD_VER > 11

template <class It>
void
test(It i)
{
    const std::reverse_iterator<It> r = std::make_reverse_iterator(i);
    assert(r.base() == i);
}

int main()
{
    const char* s = "1234567890";
    random_access_iterator<const char*>b(s);
    random_access_iterator<const char*>e(s+10);
    while ( b != e )
        test ( b++ );
}
#else
int main () {}
#endif

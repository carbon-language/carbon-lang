//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

#include "../../min_allocator.h"

int main()
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
#if __cplusplus >= 201103L
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
}

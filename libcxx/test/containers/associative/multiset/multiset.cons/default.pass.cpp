//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// multiset();

#include <set>
#include <cassert>

#include "../../../min_allocator.h"

int main()
{
    {
    std::multiset<int> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#if __cplusplus >= 201103L
    {
    std::multiset<int, std::less<int>, min_allocator<int>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#endif
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// multimap();

#include <map>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
    std::multimap<int, double> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#if __cplusplus >= 201103L
    {
    std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
    }
#endif
}

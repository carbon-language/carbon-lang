//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void merge(list& x);
// If (&addressof(x) == this) does nothing; otherwise ...

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    int a1[] = {1, 3, 7, 9, 10};
    int a2[] = {0, 2, 4, 5, 6, 8, 11};
    int a3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::list<int> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
    std::list<int> c2(a2, a2+sizeof(a2)/sizeof(a2[0]));
    c1.merge(c2);
    assert(c1 == std::list<int>(a3, a3+sizeof(a3)/sizeof(a3[0])));
    assert(c2.empty());
    }

    {
    int a1[] = {1, 3, 7, 9, 10};
    std::list<int> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c1.merge(c1);
    assert((c1 == std::list<int>(a1, a1+sizeof(a1)/sizeof(a1[0]))));
    }

#if TEST_STD_VER >= 11
    {
    int a1[] = {1, 3, 7, 9, 10};
    int a2[] = {0, 2, 4, 5, 6, 8, 11};
    int a3[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::list<int, min_allocator<int>> c1(a1, a1+sizeof(a1)/sizeof(a1[0]));
    std::list<int, min_allocator<int>> c2(a2, a2+sizeof(a2)/sizeof(a2[0]));
    c1.merge(c2);
    assert((c1 == std::list<int, min_allocator<int>>(a3, a3+sizeof(a3)/sizeof(a3[0]))));
    assert(c2.empty());
    }
#endif

  return 0;
}

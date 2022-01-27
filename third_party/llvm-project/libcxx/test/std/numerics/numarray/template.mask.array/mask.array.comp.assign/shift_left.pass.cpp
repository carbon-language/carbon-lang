//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template <class T> class mask_array

// void operator<<=(const valarray<value_type>& v) const;

#include <valarray>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int a1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::size_t N1 = sizeof(a1)/sizeof(a1[0]);
    int a2[] = {1, 2, 3, 4, 5};
    const std::size_t N2 = sizeof(a2)/sizeof(a2[0]);
    bool b[N1] = {true,  false, false, true,  true,  false,
                  false, true,  false, false, false, true};
    std::valarray<int> v1(a1, N1);
    std::valarray<int> v2(a2, N2);
    assert(N2 == std::count(b, b+N1, true));
    std::valarray<bool> vb(b, N1);
    v1[vb] <<= v2;
    assert(v1.size() == 16);
    assert(v1[ 0] ==  0);
    assert(v1[ 1] ==  1);
    assert(v1[ 2] ==  2);
    assert(v1[ 3] == 12);
    assert(v1[ 4] == 32);
    assert(v1[ 5] ==  5);
    assert(v1[ 6] ==  6);
    assert(v1[ 7] == 112);
    assert(v1[ 8] ==  8);
    assert(v1[ 9] ==  9);
    assert(v1[10] == 10);
    assert(v1[11] == 352);
    assert(v1[12] == 12);
    assert(v1[13] == 13);
    assert(v1[14] == 14);
    assert(v1[15] == 15);

  return 0;
}

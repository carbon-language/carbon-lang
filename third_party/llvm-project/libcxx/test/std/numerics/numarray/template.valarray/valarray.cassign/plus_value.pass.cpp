//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray& operator+=(const value_type& x);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        T a1[] = {1,  2,  3,  4,  5};
        T a2[] = {4,  5,  6,  7,  8};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v2(a2, N);
        v1 += 3;
        assert(v1.size() == v2.size());
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == v2[i]);
    }

  return 0;
}

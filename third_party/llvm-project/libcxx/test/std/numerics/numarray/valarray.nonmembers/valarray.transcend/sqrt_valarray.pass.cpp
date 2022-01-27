//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// template<class T>
//   valarray<T>
//   sqrt(const valarray<T>& x);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "valarray_helper.h"

int main(int, char**)
{
    {
        typedef double T;
        T a1[] = {.5, .75, 1, 3, 7};
        T a3[] = {7.0710678118654757e-01,
                  8.6602540378443860e-01,
                  1.0000000000000000e+00,
                  1.7320508075688772e+00,
                  2.6457513110645907e+00};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v3 = sqrt(v1);
        assert(v3.size() == v1.size());
        for (std::size_t i = 0; i < v3.size(); ++i)
            assert(is_about(v3[i], a3[i], 10));
    }

  return 0;
}

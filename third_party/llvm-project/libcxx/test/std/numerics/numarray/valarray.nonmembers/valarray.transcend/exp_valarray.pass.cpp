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
//   exp(const valarray<T>& x);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "valarray_helper.h"

int main(int, char**)
{
    {
        typedef double T;
        T a1[] = {-.9, -.5, 0., .5, .75};
        T a3[] = {4.0656965974059911e-01,
                  6.0653065971263342e-01,
                  1.0000000000000000e+00,
                  1.6487212707001282e+00,
                  2.1170000166126748e+00};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v3 = exp(v1);
        assert(v3.size() == v1.size());
        for (std::size_t i = 0; i < v3.size(); ++i)
            assert(is_about(v3[i], a3[i], 10));
    }

  return 0;
}

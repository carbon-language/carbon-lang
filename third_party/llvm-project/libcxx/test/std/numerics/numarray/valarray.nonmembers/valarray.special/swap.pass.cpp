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
//   void
//   swap(valarray<T>& x, valarray<T>& y);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5};
        T a2[] = {6, 7, 8, 9, 10, 11, 12};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        const unsigned N2 = sizeof(a2)/sizeof(a2[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2(a2, N2);
        std::valarray<T> v1_save = v1;
        std::valarray<T> v2_save = v2;
        swap(v1, v2);
        assert(v1.size() == v2_save.size());
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == v2_save[i]);
        assert(v2.size() == v1_save.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v1_save[i]);
    }
    {
        typedef int T;
        T a1[] = {1, 2, 3, 4, 5};
        const unsigned N1 = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N1);
        std::valarray<T> v2;
        std::valarray<T> v1_save = v1;
        std::valarray<T> v2_save = v2;
        swap(v1, v2);
        assert(v1.size() == v2_save.size());
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == v2_save[i]);
        assert(v2.size() == v1_save.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v1_save[i]);
    }
    {
        typedef int T;
        T a2[] = {6, 7, 8, 9, 10, 11, 12};
        const unsigned N2 = sizeof(a2)/sizeof(a2[0]);
        std::valarray<T> v1;
        std::valarray<T> v2(a2, N2);
        std::valarray<T> v1_save = v1;
        std::valarray<T> v2_save = v2;
        swap(v1, v2);
        assert(v1.size() == v2_save.size());
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == v2_save[i]);
        assert(v2.size() == v1_save.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v1_save[i]);
    }
    {
        typedef int T;
        std::valarray<T> v1;
        std::valarray<T> v2;
        std::valarray<T> v1_save = v1;
        std::valarray<T> v2_save = v2;
        swap(v1, v2);
        assert(v1.size() == v2_save.size());
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == v2_save[i]);
        assert(v2.size() == v1_save.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v1_save[i]);
    }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray& operator=(const valarray& v);

#include <valarray>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

struct S
{
    S() : x_(0) { default_ctor_called = true; }
    S(int x) : x_(x) {}
    int x_;
    static bool default_ctor_called;
};

bool S::default_ctor_called = false;

bool operator==(const S& lhs, const S& rhs)
{
    return lhs.x_ == rhs.x_;
}

int main(int, char**)
{
    {
        typedef int T;
        T a[] = {1, 2, 3, 4, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2;
        v2 = v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v[i]);
    }
    {
        typedef double T;
        T a[] = {1, 2.5, 3, 4.25, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2;
        v2 = v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v[i]);
    }
    {
        typedef std::valarray<double> T;
        T a[] = {T(1), T(2), T(3), T(4), T(5)};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2(a, N-2);
        v2 = v;
        assert(v2.size() == v.size());
        for (unsigned i = 0; i < N; ++i)
        {
            assert(v2[i].size() == v[i].size());
            for (std::size_t j = 0; j < v[i].size(); ++j)
                assert(v2[i][j] == v[i][j]);
        }
    }
    {
        typedef S T;
        T a[] = {T(1), T(2), T(3), T(4), T(5)};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v(a, N);
        std::valarray<T> v2;
        v2 = v;
        assert(v2.size() == v.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == v[i]);
        assert(!S::default_ctor_called);
    }

  return 0;
}

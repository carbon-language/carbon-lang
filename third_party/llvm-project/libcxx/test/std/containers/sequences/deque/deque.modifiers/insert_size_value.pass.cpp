//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// <deque>

// iterator insert (const_iterator p, size_type n, const value_type& v);

#include <deque>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "min_allocator.h"

template <class C>
C
make(int size, int start = 0 )
{
    const int b = 4096 / sizeof(int);
    int init = 0;
    if (start > 0)
    {
        init = (start+1) / b + ((start+1) % b != 0);
        init *= b;
        --init;
    }
    C c(init, 0);
    for (int i = 0; i < init-start; ++i)
        c.pop_back();
    for (int i = 0; i < size; ++i)
        c.push_back(i);
    for (int i = 0; i < start; ++i)
        c.pop_front();
    return c;
}

template <class C>
void
test(int P, C& c1, int size, int x)
{
    typedef typename C::const_iterator CI;
    std::size_t c1_osize = c1.size();
    CI i = c1.insert(c1.begin() + P, size, x);
    assert(i == c1.begin() + P);
    assert(c1.size() == c1_osize + size);
    assert(static_cast<std::size_t>(std::distance(c1.begin(), c1.end())) == c1.size());
    i = c1.begin();
    for (int j = 0; j < P; ++j, ++i)
        assert(*i == j);
    for (int j = 0; j < size; ++j, ++i)
        assert(*i == x);
    for (int j = P; static_cast<std::size_t>(j) < c1_osize; ++j, ++i)
        assert(*i == j);
}

template <class C>
void
testN(int start, int N, int M)
{
    for (int i = 0; i <= 3; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make<C>(N, start);
            test(i, c1, M, -10);
        }
    }
    for (int i = M-1; i <= M+1; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make<C>(N, start);
            test(i, c1, M, -10);
        }
    }
    for (int i = N/2-1; i <= N/2+1; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make<C>(N, start);
            test(i, c1, M, -10);
        }
    }
    for (int i = N - M - 1; i <= N - M + 1; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make<C>(N, start);
            test(i, c1, M, -10);
        }
    }
    for (int i = N - 3; i <= N; ++i)
    {
        if (0 <= i && i <= N)
        {
            C c1 = make<C>(N, start);
            test(i, c1, M, -10);
        }
    }
}

template <class C>
void
self_reference_test()
{
    typedef typename C::const_iterator CI;
    for (int i = 0; i < 20; ++i)
    {
        for (int j = 0; j < 20; ++j)
        {
            C c = make<C>(20);
            CI it = c.cbegin() + i;
            CI jt = c.cbegin() + j;
            c.insert(it, 5, *jt);
            assert(c.size() == 25);
            assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
            it = c.cbegin();
            for (int k = 0; k < i; ++k, ++it)
                assert(*it == k);
            for (int k = 0; k < 5; ++k, ++it)
                assert(*it == j);
            for (int k = i; k < 20; ++k, ++it)
                assert(*it == k);
        }
    }
}

int main(int, char**)
{
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                testN<std::deque<int> >(rng[i], rng[j], rng[k]);
    self_reference_test<std::deque<int> >();
    }
#if TEST_STD_VER >= 11
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                testN<std::deque<int, min_allocator<int>> >(rng[i], rng[j], rng[k]);
    self_reference_test<std::deque<int, min_allocator<int>> >();
    }
#endif

  return 0;
}

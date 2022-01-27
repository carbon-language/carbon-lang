//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Optimization for deque::iterators

// template <class InputIterator, class OutputIterator>
//   OutputIterator
//   move_backward(InputIterator first, InputIterator last, OutputIterator result);

#include <deque>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
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
void testN(int start, int N)
{
    typedef typename C::iterator I;
    typedef typename C::const_iterator CI;
    typedef random_access_iterator<I> RAI;
    typedef random_access_iterator<CI> RACI;
    C c1 = make<C>(N, start);
    C c2 = make<C>(N);
    assert(std::move_backward(c1.cbegin(), c1.cend(), c2.end()) == c2.begin());
    assert(c1 == c2);
    assert(std::move_backward(c2.cbegin(), c2.cend(), c1.end()) == c1.begin());
    assert(c1 == c2);
    assert(std::move_backward(c1.cbegin(), c1.cend(), RAI(c2.end())) == RAI(c2.begin()));
    assert(c1 == c2);
    assert(std::move_backward(c2.cbegin(), c2.cend(), RAI(c1.end())) == RAI(c1.begin()));
    assert(c1 == c2);
    assert(std::move_backward(RACI(c1.cbegin()), RACI(c1.cend()), c2.end()) == c2.begin());
    assert(c1 == c2);
    assert(std::move_backward(RACI(c2.cbegin()), RACI(c2.cend()), c1.end()) == c1.begin());
    assert(c1 == c2);
}

int main(int, char**)
{
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN<std::deque<int> >(rng[i], rng[j]);
    }
#if TEST_STD_VER >= 11
    {
    int rng[] = {0, 1, 2, 3, 1023, 1024, 1025, 2047, 2048, 2049};
    const int N = sizeof(rng)/sizeof(rng[0]);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            testN<std::deque<int, min_allocator<int> > >(rng[i], rng[j]);
    }
#endif

  return 0;
}

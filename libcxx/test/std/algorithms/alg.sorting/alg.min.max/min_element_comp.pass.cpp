//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires CopyConstructible<Compare>
//   Iter
//   min_element(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <random>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

std::mt19937 randomness;

template <class Iter>
void
test(Iter first, Iter last)
{
    Iter i = std::min_element(first, last, std::greater<int>());
    if (first != last)
    {
        for (Iter j = first; j != last; ++j)
            assert(!std::greater<int>()(*j, *i));
    }
    else
        assert(i == last);
}

template <class Iter>
void
test(int N)
{
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = i;
    std::shuffle(a, a+N, randomness);
    test(Iter(a), Iter(a+N));
    delete [] a;
}

template <class Iter>
void
test()
{
    test<Iter>(0);
    test<Iter>(1);
    test<Iter>(2);
    test<Iter>(3);
    test<Iter>(10);
    test<Iter>(1000);
}

template <class Iter, class Pred>
void test_eq0(Iter first, Iter last, Pred p)
{
    assert(first == std::min_element(first, last, p));
}

void test_eq()
{
    const int N = 10;
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = 10; // all the same
    test_eq0(a, a+N, std::less<int>());
    test_eq0(a, a+N, std::greater<int>());
    delete [] a;
}

#if TEST_STD_VER >= 14
constexpr int il[] = { 2, 4, 6, 8, 7, 5, 3, 1 };
struct less { constexpr bool operator ()( const int &x, const int &y) const { return x < y; }};
#endif

void constexpr_test()
{
#if TEST_STD_VER >= 14
    constexpr auto p = std::min_element(il, il+8, less());
    static_assert(*p == 1, "");
#endif
}

int main(int, char**)
{
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test_eq();

    constexpr_test();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class OutputIterator, class T>
//     OutputIterator inclusive_scan(InputIterator first, InputIterator last,
//                                   OutputIterator result, T init);
//

#include <numeric>
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class T>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first, Iter1 last, const T *rFirst, const T *rLast)
{
    assert((rLast - rFirst) <= 5);  // or else increase the size of "out"
    T out[5];

    // Not in place
    T *end = std::inclusive_scan(first, last, out);
    assert(std::equal(out, end, rFirst, rLast));

    // In place
    std::copy(first, last, out);
    end = std::inclusive_scan(out, end, out);
    assert(std::equal(out, end, rFirst, rLast));
}


template <class Iter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[]         = {1, 3, 5, 7,  9};
    const int pRes[] = {1, 4, 9, 16, 25};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    static_assert(sa == sizeof(pRes) / sizeof(pRes[0]));       // just to be sure

    for (unsigned int i = 0; i < sa; ++i ) {
        test(Iter(ia), Iter(ia + i), pRes, pRes + i);
    }
}

constexpr size_t triangle(size_t n) { return n*(n+1)/2; }

//  Basic sanity
TEST_CONSTEXPR_CXX20 void
basic_tests()
{
    {
    std::array<size_t, 10> v;
    std::fill(v.begin(), v.end(), 3);
    std::inclusive_scan(v.begin(), v.end(), v.begin());
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == (i+1) * 3);
    }

    {
    std::array<size_t, 10> v;
    std::iota(v.begin(), v.end(), 0);
    std::inclusive_scan(v.begin(), v.end(), v.begin());
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == triangle(i));
    }

    {
    std::array<size_t, 10> v;
    std::iota(v.begin(), v.end(), 1);
    std::inclusive_scan(v.begin(), v.end(), v.begin());
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == triangle(i + 1));
    }

    {
    std::array<size_t, 0> v, res;
    std::inclusive_scan(v.begin(), v.end(), res.begin());
    assert(res.empty());
    }
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    basic_tests();

//  All the iterator categories
    test<cpp17_input_iterator        <const int*> >();
    test<forward_iterator      <const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test<      int*>();

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    return 0;
}

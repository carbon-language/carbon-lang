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
// template<class InputIterator, class OutputIterator, class T,
//          class BinaryOperation, class UnaryOperation>
//   OutputIterator transform_inclusive_scan(InputIterator first, InputIterator last,
//                                           OutputIterator result,
//                                           BinaryOperation binary_op,
//                                           UnaryOperation unary_op,
//                                           T init);


#include <numeric>
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

struct add_one {
    template <typename T>
    constexpr T operator()(T x) const {
        return x + 1;
    }
};

template <class Iter1, class BOp, class UOp, class T>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first, Iter1 last, BOp bop, UOp uop, T init, const T *rFirst, const T *rLast)
{
    assert((rLast - rFirst) <= 5);  // or else increase the size of "out"
    T out[5];

    // Not in place
    T *end = std::transform_inclusive_scan(first, last, out, bop, uop, init);
    assert(std::equal(out, end, rFirst, rLast));

    // In place
    std::copy(first, last, out);
    end = std::transform_inclusive_scan(out, end, out, bop, uop, init);
    assert(std::equal(out, end, rFirst, rLast));
}


template <class Iter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[]           = {  1,  3,   5,    7,     9 };
    const int pResI0[] = {  2,  6,  12,   20,    30 };        // with add_one
    const int mResI0[] = {  0,  0,   0,    0,     0 };
    const int pResN0[] = { -1, -4,  -9,  -16,   -25 };        // with negate
    const int mResN0[] = {  0,  0,   0,    0,     0 };
    const int pResI2[] = {  4,  8,  14,   22,    32 };        // with add_one
    const int mResI2[] = {  4, 16,  96,  768,  7680 };
    const int pResN2[] = {  1, -2,  -7,  -14,   -23 };        // with negate
    const int mResN2[] = { -2,  6, -30,  210, -1890 };
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    static_assert(sa == sizeof(pResI0) / sizeof(pResI0[0]));       // just to be sure
    static_assert(sa == sizeof(mResI0) / sizeof(mResI0[0]));       // just to be sure
    static_assert(sa == sizeof(pResN0) / sizeof(pResN0[0]));       // just to be sure
    static_assert(sa == sizeof(mResN0) / sizeof(mResN0[0]));       // just to be sure
    static_assert(sa == sizeof(pResI2) / sizeof(pResI2[0]));       // just to be sure
    static_assert(sa == sizeof(mResI2) / sizeof(mResI2[0]));       // just to be sure
    static_assert(sa == sizeof(pResN2) / sizeof(pResN2[0]));       // just to be sure
    static_assert(sa == sizeof(mResN2) / sizeof(mResN2[0]));       // just to be sure

    for (unsigned int i = 0; i < sa; ++i ) {
        test(Iter(ia), Iter(ia + i), std::plus<>(),       add_one{},       0, pResI0, pResI0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), add_one{},       0, mResI0, mResI0 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       std::negate<>(), 0, pResN0, pResN0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), std::negate<>(), 0, mResN0, mResN0 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       add_one{},       2, pResI2, pResI2 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), add_one{},       2, mResI2, mResI2 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       std::negate<>(), 2, pResN2, pResN2 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), std::negate<>(), 2, mResN2, mResN2 + i);
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
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), add_one{}, size_t{50});
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 50 + (i + 1) * 4);
    }

    {
    std::array<size_t, 10> v;
    std::iota(v.begin(), v.end(), 0);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), add_one{}, size_t{30});
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 30 + triangle(i) + i + 1);
    }

    {
    std::array<size_t, 10> v;
    std::iota(v.begin(), v.end(), 1);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), add_one{}, size_t{40});
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 40 + triangle(i + 1) + i + 1);
    }

    {
    std::array<size_t, 0> v, res;
    std::transform_inclusive_scan(v.begin(), v.end(), res.begin(), std::plus<>(), add_one{}, size_t{1});
    assert(res.empty());
    }

//  Make sure that the calculations are done using the init typedef
    {
    std::array<unsigned char, 10> v;
    std::iota(v.begin(), v.end(), static_cast<unsigned char>(1));
    std::array<size_t, 10> res;
    std::transform_inclusive_scan(v.begin(), v.end(), res.begin(), std::multiplies<>(), add_one{}, size_t{1});

    assert(res.size() == 10);
    size_t j = 2;
    assert(res[0] == 2);
    for (size_t i = 1; i < res.size(); ++i)
    {
        j *= i + 2;
        assert(res[i] == j);
    }
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

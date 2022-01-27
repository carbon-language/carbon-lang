//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests
//
// This fails on MinGW for some unknown reason.
// UNSUPPORTED: target={{.+}}-windows-gnu

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <cassert>
#include <climits>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "test_macros.h"

template <class T>
T sqr(T x)
{
    return x * x;
}

template <class ResultType, class EngineType>
void test_statistics(ResultType a, ResultType b)
{
    ASSERT_SAME_TYPE(typename std::uniform_int_distribution<ResultType>::result_type, ResultType);

    EngineType g;
    std::uniform_int_distribution<ResultType> dist(a, b);
    assert(dist.a() == a);
    assert(dist.b() == b);
    std::vector<ResultType> u;
    for (int i = 0; i < 10000; ++i) {
        ResultType v = dist(g);
        assert(a <= v && v <= b);
        u.push_back(v);
    }

    // Quick check: The chance of getting *no* hits in any given tenth of the range
    // is (0.9)^10000, or "ultra-astronomically low."
    bool bottom_tenth = false;
    bool top_tenth = false;
    for (std::size_t i = 0; i < u.size(); ++i) {
        bottom_tenth = bottom_tenth || (u[i] <= (a + (b / 10) - (a / 10)));
        top_tenth = top_tenth || (u[i] >= (b - (b / 10) + (a / 10)));
    }
    assert(bottom_tenth);  // ...is populated
    assert(top_tenth);  // ...is populated

    // Now do some more involved statistical math.
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        double dbl = (u[i] - mean);
        double d2 = dbl * dbl;
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;

    double expected_mean = double(a) + double(b)/2 - double(a)/2;
    double expected_var = (sqr(double(b) - double(a) + 1) - 1) / 12;

    double range = double(b) - double(a) + 1.0;
    assert(range > range / 10);  // i.e., it's not infinity

    assert(std::abs(mean - expected_mean) < range / 100);
    assert(std::abs(var - expected_var) < expected_var / 50);
    assert(-0.1 < skew && skew < 0.1);
    assert(1.6 < kurtosis && kurtosis < 2.0);
}

template <class ResultType, class EngineType>
void test_statistics()
{
    test_statistics<ResultType, EngineType>(0, std::numeric_limits<ResultType>::max());
}

int main(int, char**)
{
    test_statistics<int, std::minstd_rand0>();
    test_statistics<int, std::minstd_rand>();
    test_statistics<int, std::mt19937>();
    test_statistics<int, std::mt19937_64>();
    test_statistics<int, std::ranlux24_base>();
    test_statistics<int, std::ranlux48_base>();
    test_statistics<int, std::ranlux24>();
    test_statistics<int, std::ranlux48>();
    test_statistics<int, std::knuth_b>();
    test_statistics<int, std::minstd_rand0>(-6, 106);
    test_statistics<int, std::minstd_rand>(5, 100);

    test_statistics<short, std::minstd_rand0>();
    test_statistics<int, std::minstd_rand0>();
    test_statistics<long, std::minstd_rand0>();
    test_statistics<long long, std::minstd_rand0>();

    test_statistics<unsigned short, std::minstd_rand0>();
    test_statistics<unsigned int, std::minstd_rand0>();
    test_statistics<unsigned long, std::minstd_rand0>();
    test_statistics<unsigned long long, std::minstd_rand0>();

    test_statistics<short, std::minstd_rand0>(SHRT_MIN, SHRT_MAX);

    // http://eel.is/c++draft/rand.req#genl-1.5
    // The effect of instantiating a template that has a parameter
    // named IntType is undefined unless the corresponding template
    // argument is cv-unqualified and is one of short, int, long,
    // long long, unsigned short, unsigned int, unsigned long,
    // or unsigned long long.
    // (We support __int128 as an extension.)

#ifndef _LIBCPP_HAS_NO_INT128
    test_statistics<__int128_t, std::minstd_rand0>();
    test_statistics<__uint128_t, std::minstd_rand0>();

    test_statistics<__int128_t, std::minstd_rand0>(-100, 900);
    test_statistics<__int128_t, std::minstd_rand0>(0, UINT64_MAX);
    test_statistics<__int128_t, std::minstd_rand0>(std::numeric_limits<__int128_t>::min(), std::numeric_limits<__int128_t>::max());
    test_statistics<__uint128_t, std::minstd_rand0>(0, UINT64_MAX);
#endif

    return 0;
}

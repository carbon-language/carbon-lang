//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <numeric>
#include <vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <class T>
inline
T
sqr(T x)
{
    return x * x;
}

int main(int, char**)
{
    {
        typedef std::bernoulli_distribution D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(.75);
        P p(.25);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            double dbl = (u[i] - mean);
            double d2 = sqr(dbl);
            var += d2;
            skew += dbl * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = p.p();
        double x_var = p.p()*(1-p.p());
        double x_skew = (1 - 2 * p.p())/std::sqrt(x_var);
        double x_kurtosis = (6 * sqr(p.p()) - 6 * p.p() + 1)/x_var;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
    }
    {
        typedef std::bernoulli_distribution D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(.25);
        P p(.75);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            double dbl = (u[i] - mean);
            double d2 = sqr(dbl);
            var += d2;
            skew += dbl * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = p.p();
        double x_var = p.p()*(1-p.p());
        double x_skew = (1 - 2 * p.p())/std::sqrt(x_var);
        double x_kurtosis = (6 * sqr(p.p()) - 6 * p.p() + 1)/x_var;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
    }

  return 0;
}

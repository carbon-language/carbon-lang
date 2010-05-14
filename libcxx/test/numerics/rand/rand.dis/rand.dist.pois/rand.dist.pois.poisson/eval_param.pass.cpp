//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class poisson_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <cassert>
#include <vector>
#include <numeric>

template <class T>
inline
T
sqr(T x)
{
    return x * x;
}

int main()
{
    {
        typedef std::poisson_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(.75);
        P p(2);
        const int N = 100000;
        std::vector<double> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = p.mean();
        double x_var = p.mean();
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::poisson_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(2);
        P p(.75);
        const int N = 100000;
        std::vector<double> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = p.mean();
        double x_var = p.mean();
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::poisson_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(2);
        P p(20);
        const int N = 10000;
        std::vector<double> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = p.mean();
        double x_var = p.mean();
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
}

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
// class binomial_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <numeric>
#include <vector>
#include <cassert>

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
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(5, .75);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(30, .03125);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(40, .25);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(40, 0);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(mean == x_mean);
        assert(var == x_var);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(40, 1);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(mean == x_mean);
        assert(var == x_var);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(400, 0.5);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(1, 0.5);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(std::abs(mean - x_mean) / x_mean < 0.01);
        assert(std::abs(var - x_var) / x_var < 0.01);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(0, 0.005);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(mean == x_mean);
        assert(var == x_var);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(0, 0);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(mean == x_mean);
        assert(var == x_var);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef std::minstd_rand G;
        G g;
        D d(0, 1);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        double mean = std::accumulate(u.begin(), u.end(),
                                              double(0)) / u.size();
        double var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        double x_mean = d.t() * d.p();
        double x_var = x_mean*(1-d.p());
        assert(mean == x_mean);
        assert(var == x_var);
    }
}

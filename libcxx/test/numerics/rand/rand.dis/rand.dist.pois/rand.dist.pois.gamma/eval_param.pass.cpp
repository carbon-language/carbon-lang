//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class gamma_distribution

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
        typedef std::gamma_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(0.5, 2);
        P p(1, .5);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        D::result_type mean = std::accumulate(u.begin(), u.end(),
                                              D::result_type(0)) / u.size();
        D::result_type var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        D::result_type x_mean = p.alpha() * p.beta();
        D::result_type x_var = p.alpha() * sqr(p.beta());
        assert(std::abs(mean - x_mean) / x_mean < 0.02);
        assert(std::abs(var - x_var) / x_var < 0.02);
    }
    {
        typedef std::gamma_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(1, .5);
        P p(2, 3);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        D::result_type mean = std::accumulate(u.begin(), u.end(),
                                              D::result_type(0)) / u.size();
        D::result_type var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        D::result_type x_mean = p.alpha() * p.beta();
        D::result_type x_var = p.alpha() * sqr(p.beta());
        assert(std::abs(mean - x_mean) / x_mean < 0.02);
        assert(std::abs(var - x_var) / x_var < 0.02);
    }
    {
        typedef std::gamma_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d(2, 3);
        P p(.5, 2);
        const int N = 100000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g, p));
        D::result_type mean = std::accumulate(u.begin(), u.end(),
                                              D::result_type(0)) / u.size();
        D::result_type var = 0;
        for (int i = 0; i < u.size(); ++i)
            var += sqr(u[i] - mean);
        var /= u.size();
        D::result_type x_mean = p.alpha() * p.beta();
        D::result_type x_var = p.alpha() * sqr(p.beta());
        assert(std::abs(mean - x_mean) / x_mean < 0.02);
        assert(std::abs(var - x_var) / x_var < 0.02);
    }
}

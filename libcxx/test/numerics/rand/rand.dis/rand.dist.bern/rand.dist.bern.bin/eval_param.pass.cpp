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

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand0 G;
        G g;
        D d(16, .75);
        P p(16, .25);
        int count = 0;
        int r = 0;
        for (int i = 0; i < 100; ++i)
        {
            D::result_type u = d(g, p);
            r += u;
        }
        assert(int(r/100. + .5) == 4);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand0 G;
        G g;
        D d(16, .75);
        P p(16, .5);
        int count = 0;
        int r = 0;
        for (int i = 0; i < 100; ++i)
        {
            D::result_type u = d(g, p);
            r += u;
        }
        assert(int(r/100. + .5) == 8);
    }
    {
        typedef std::binomial_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand0 G;
        G g;
        D d(16, .75);
        P p(16, .75);
        int count = 0;
        int r = 0;
        for (int i = 0; i < 100; ++i)
        {
            D::result_type u = d(g, p);
            r += u;
        }
        assert(int(r/100. + .5) == 12);
    }
}

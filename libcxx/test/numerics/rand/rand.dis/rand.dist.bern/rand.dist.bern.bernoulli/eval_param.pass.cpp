//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::bernoulli_distribution D;
        typedef D::param_type P;
        typedef std::minstd_rand0 G;
        G g;
        D d(.75);
        P p(.25);
        int count = 0;
        for (int i = 0; i < 10000; ++i)
        {
            bool u = d(g, p);
            if (u)
                ++count;
        }
        assert(count < 2600);
    }
}

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
// class uniform_real_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::uniform_real_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand0 G;
        G g;
        D d(-6, 106);
        P p(-10, 20);
        for (int i = 0; i < 10000; ++i)
        {
            int u = d(g, p);
            assert(p.a() <= u && u < p.b());
        }
    }
}

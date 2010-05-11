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

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <cassert>
#include <iostream>

int main()
{
    {
        typedef std::uniform_real_distribution<> D;
        typedef std::minstd_rand0 G;
        G g;
        D d(-6.5, 106.75);
        for (int i = 0; i < 10000; ++i)
        {
            D::result_type u = d(g);
            assert(d.min() <= u && u < d.max());
        }
    }
}

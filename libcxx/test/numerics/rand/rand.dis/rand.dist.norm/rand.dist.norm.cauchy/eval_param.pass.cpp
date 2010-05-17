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
// class cauchy_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>

int main()
{
    typedef std::cauchy_distribution<> D;
    typedef D::param_type P;
    typedef std::mt19937 G;
    G g;
    D d(0.5, 2);
    P p(3, 4);
    D::result_type v = d(g, p);

// If anyone can figure out a better test than this,
// it would be more than welcome!
}

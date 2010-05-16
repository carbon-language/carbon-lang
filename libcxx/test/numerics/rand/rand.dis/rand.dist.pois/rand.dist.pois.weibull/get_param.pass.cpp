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
// class weibull_distribution

// param_type param() const;

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::weibull_distribution<> D;
        typedef D::param_type P;
        P p(.125, .5);
        D d(p);
        assert(d.param() == p);
    }
}

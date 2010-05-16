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

// bool operator=(const weibull_distribution& x,
//                const weibull_distribution& y);
// bool operator!(const weibull_distribution& x,
//                const weibull_distribution& y);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::weibull_distribution<> D;
        D d1(2.5, 4);
        D d2(2.5, 4);
        assert(d1 == d2);
    }
    {
        typedef std::weibull_distribution<> D;
        D d1(2.5, 4);
        D d2(2.5, 4.5);
        assert(d1 != d2);
    }
}

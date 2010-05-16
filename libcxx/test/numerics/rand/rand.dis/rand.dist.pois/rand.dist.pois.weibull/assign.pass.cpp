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

// weibull_distribution& operator=(const weibull_distribution&);

#include <random>
#include <cassert>

void
test1()
{
    typedef std::weibull_distribution<> D;
    D d1(20, 0.75);
    D d2;
    assert(d1 != d2);
    d2 = d1;
    assert(d1 == d2);
}

int main()
{
    test1();
}

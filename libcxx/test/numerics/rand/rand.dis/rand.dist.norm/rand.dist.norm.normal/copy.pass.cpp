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
// class normal_distribution

// normal_distribution(const normal_distribution&);

#include <random>
#include <cassert>

void
test1()
{
    typedef std::normal_distribution<> D;
    D d1(20, 1.75);
    D d2 = d1;
    assert(d1 == d2);
}

int main()
{
    test1();
}

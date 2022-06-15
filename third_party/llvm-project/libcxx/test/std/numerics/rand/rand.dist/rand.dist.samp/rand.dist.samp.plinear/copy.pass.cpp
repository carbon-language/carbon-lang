//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_linear_distribution

// piecewise_linear_distribution(const piecewise_linear_distribution&);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    typedef std::piecewise_linear_distribution<> D;
    double p[] = {2, 4, 1, 8, 2};
    double b[] = {2, 4, 5, 8, 9};
    D d1(b, b+5, p);
    D d2 = d1;
    assert(d1 == d2);
}

int main(int, char**)
{
    test1();

  return 0;
}

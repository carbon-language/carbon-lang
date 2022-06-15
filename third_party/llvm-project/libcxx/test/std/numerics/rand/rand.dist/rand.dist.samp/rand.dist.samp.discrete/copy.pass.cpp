//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class discrete_distribution

// discrete_distribution(const discrete_distribution&);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    typedef std::discrete_distribution<> D;
    double p[] = {2, 4, 1, 8};
    D d1(p, p+4);
    D d2 = d1;
    assert(d1 == d2);
}

int main(int, char**)
{
    test1();

  return 0;
}

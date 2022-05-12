//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class uniform_real_distribution

// uniform_real_distribution(const uniform_real_distribution&);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    typedef std::uniform_real_distribution<float> D;
    D d1(2, 5);
    D d2 = d1;
    assert(d1 == d2);
}

int main(int, char**)
{
    test1();

  return 0;
}

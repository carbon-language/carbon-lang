//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, size_t w, size_t n, size_t m, size_t r,
//           UIntType a, size_t u, UIntType d, size_t s,
//           UIntType b, size_t t, UIntType c, size_t l, UIntType f>
// class mersenne_twister_engine;

// void discard(unsigned long long z);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    std::mt19937 e1;
    std::mt19937 e2 = e1;
    assert(e1 == e2);
    e1.discard(3);
    assert(e1 != e2);
    (void)e2();
    (void)e2();
    (void)e2();
    assert(e1 == e2);
}

void
test2()
{
    std::mt19937_64 e1;
    std::mt19937_64 e2 = e1;
    assert(e1 == e2);
    e1.discard(3);
    assert(e1 != e2);
    (void)e2();
    (void)e2();
    (void)e2();
    assert(e1 == e2);
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}

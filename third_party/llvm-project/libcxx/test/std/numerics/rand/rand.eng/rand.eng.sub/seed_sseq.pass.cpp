//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class UIntType, size_t w, size_t s, size_t r>
// class subtract_with_carry_engine;

// template<class Sseq> void seed(Sseq& q);

#include <random>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    unsigned a[] = {3, 5, 7};
    std::seed_seq sseq(a, a+3);
    std::ranlux24_base e1;
    std::ranlux24_base e2(sseq);
    assert(e1 != e2);
    e1.seed(sseq);
    assert(e1 == e2);
}

void
test2()
{
    unsigned a[] = {3, 5, 7};
    std::seed_seq sseq(a, a+3);
    std::ranlux48_base e1;
    std::ranlux48_base e2(sseq);
    assert(e1 != e2);
    e1.seed(sseq);
    assert(e1 == e2);
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}

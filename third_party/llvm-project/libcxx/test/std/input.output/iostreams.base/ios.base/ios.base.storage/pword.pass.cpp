//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// void*& pword(int idx);

// This test compiles but never completes when compiled against the MSVC STL
// UNSUPPORTED: stdlib=msvc

#include <ios>
#include <string>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

class test
    : public std::ios
{
public:
    test()
    {
        init(0);
    }
};

int main(int, char**)
{
    test t;
    std::ios_base& b = t;
    for (std::intptr_t i = 0; i < 10000; ++i)
    {
        assert(b.pword(i) == 0);
        b.pword(i) = (void*)i;
        assert(b.pword(i) == (void*)i);
        for (std::intptr_t j = 0; j <= i; ++j)
            assert(b.pword(j) == (void*)j);
    }

  return 0;
}

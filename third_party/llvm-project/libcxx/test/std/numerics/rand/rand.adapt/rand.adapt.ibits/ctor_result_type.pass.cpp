//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t w, class UIntType>
// class independent_bits_engine

// explicit independent_bits_engine(result_type s = default_seed);

// Serializing/deserializing the state of the RNG requires iostreams
// UNSUPPORTED: no-localization

#include <random>
#include <sstream>
#include <cassert>

#include "test_macros.h"

void
test1()
{
    const char* a = "15136306 8587749 2346244 16479026 15515802 9510553 "
    "16090340 14501685 13839944 10789678 11581259 9590790 5840316 5953700 "
    "13398366 8134459 16629731 6851902 15583892 1317475 4231148 9092691 "
    "5707268 2355175 0 0";
    std::independent_bits_engine<std::ranlux24, 32, unsigned> e1(0);
    std::ostringstream os;
    os << e1;
    assert(os.str() == a);
}

void
test2()
{
    const char* a = "10880375256626 126660097854724 33643165434010 "
    "78293780235492 179418984296008 96783156950859 238199764491708 "
    "34339434557790 155299155394531 29014415493780 209265474179052 "
    "263777435457028 0 0";
    std::independent_bits_engine<std::ranlux48, 64, unsigned long long> e1(0);
    std::ostringstream os;
    os << e1;
    assert(os.str() == a);
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}

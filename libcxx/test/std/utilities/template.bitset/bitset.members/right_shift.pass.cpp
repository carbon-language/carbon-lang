//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N> operator>>(size_t pos) const;

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
void test_right_shift() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        for (std::size_t s = 0; s <= N+1; ++s) {
            std::bitset<N> v1 = cases[c];
            std::bitset<N> v2 = v1;
            assert((v1 >>= s) == (v2 >> s));
        }
    }
}

int main(int, char**) {
    test_right_shift<0>();
    test_right_shift<1>();
    test_right_shift<31>();
    test_right_shift<32>();
    test_right_shift<33>();
    test_right_shift<63>();
    test_right_shift<64>();
    test_right_shift<65>();
    test_right_shift<1000>();

    return 0;
}

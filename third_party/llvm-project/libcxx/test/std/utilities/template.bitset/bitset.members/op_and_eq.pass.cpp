//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bitset<N>& operator&=(const bitset<N>& rhs);

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
void test_op_and_eq() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c1 = 0; c1 != cases.size(); ++c1) {
        for (std::size_t c2 = 0; c2 != cases.size(); ++c2) {
            std::bitset<N> v1 = cases[c1];
            std::bitset<N> v2 = cases[c2];
            std::bitset<N> v3 = v1;
            v1 &= v2;
            for (std::size_t i = 0; i < v1.size(); ++i)
                assert(v1[i] == (v3[i] && v2[i]));
        }
    }
}

int main(int, char**) {
    test_op_and_eq<0>();
    test_op_and_eq<1>();
    test_op_and_eq<31>();
    test_op_and_eq<32>();
    test_op_and_eq<33>();
    test_op_and_eq<63>();
    test_op_and_eq<64>();
    test_op_and_eq<65>();
    test_op_and_eq<1000>();

    return 0;
}

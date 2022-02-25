//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// requirements on parameters

#include <random>

int main(int, char**)
{
    typedef unsigned long long T;

    // expected-error@*:* {{static_assert failed due to requirement '1ULL == 0 || 1ULL < 1ULL' "linear_congruential_engine invalid parameters"}}
    std::linear_congruential_engine<T, 0, 0, 0> e2;
    // expected-error@*:* {{static_assert failed due to requirement '1ULL == 0 || 1ULL < 1ULL' "linear_congruential_engine invalid parameters"}}
    std::linear_congruential_engine<T, 0, 1, 1> e3;
    std::linear_congruential_engine<T, 1, 0, 1> e4;
    // expected-error@*:* {{static_assert failed due to requirement 'is_unsigned<int>::value' "_UIntType must be unsigned type"}}
    std::linear_congruential_engine<int, 0, 0, 0> e5;

    return 0;
}

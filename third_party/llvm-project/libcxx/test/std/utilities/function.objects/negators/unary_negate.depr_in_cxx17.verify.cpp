//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// unary_negate
//  deprecated in C++17

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_NEGATORS

#include <functional>

#include "test_macros.h"

struct Predicate {
    typedef int argument_type;
    bool operator()(argument_type) const { return true; }
};

int main(int, char**) {
    std::unary_negate<Predicate> f((Predicate())); // expected-warning {{'unary_negate<Predicate>' is deprecated}}
    (void)f;

  return 0;
}

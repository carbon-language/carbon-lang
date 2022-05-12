//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// allocator:
// T* allocate(size_t n);

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::allocator<int> a;
    a.allocate(3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}

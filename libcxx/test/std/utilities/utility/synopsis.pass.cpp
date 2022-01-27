//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>
// XFAIL: c++03

// #include <initializer_list>

#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    std::initializer_list<int> x;
    (void)x;

  return 0;
}

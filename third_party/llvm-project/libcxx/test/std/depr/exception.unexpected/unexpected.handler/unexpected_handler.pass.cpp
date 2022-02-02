//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14

// test unexpected_handler

#include <exception>

#include "test_macros.h"

void f() {}

int main(int, char**)
{
    std::unexpected_handler p = f;
    ((void)p);

  return 0;
}

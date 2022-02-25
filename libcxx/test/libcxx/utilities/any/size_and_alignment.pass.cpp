//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// Check that the size and alignment of any are what we expect.

#include <any>

#include "test_macros.h"

int main(int, char**)
{
    static_assert(sizeof(std::any) == sizeof(void*)*4, "");
    static_assert(alignof(std::any) == alignof(void*), "");

  return 0;
}

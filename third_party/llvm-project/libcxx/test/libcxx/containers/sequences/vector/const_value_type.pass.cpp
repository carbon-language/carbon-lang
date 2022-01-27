//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// vector<const int> v;  // an extension

#include <vector>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    std::vector<const int> v = {1, 2, 3};

  return 0;
}

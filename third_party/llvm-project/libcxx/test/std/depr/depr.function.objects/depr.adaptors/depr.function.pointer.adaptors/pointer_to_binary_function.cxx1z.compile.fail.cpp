//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// pointer_to_binary_function
// UNSUPPORTED: c++03, c++11, c++14

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

double binary_f(int i, short j) {return i - j + .75;}

int main(int, char**)
{
    typedef std::pointer_to_binary_function<int, short, double> F;

  return 0;
}

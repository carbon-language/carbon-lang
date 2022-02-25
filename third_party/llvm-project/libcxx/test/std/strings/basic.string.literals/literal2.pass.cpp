// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using namespace std::literals::string_literals;

    std::string foo  =   ""s;

  return 0;
}

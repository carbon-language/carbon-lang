//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <condition_variable>

// class condition_variable;

// condition_variable();

#include <condition_variable>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::condition_variable cv;
    static_cast<void>(cv);

  return 0;
}

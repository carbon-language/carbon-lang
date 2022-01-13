//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// void declare_reachable(void* p);
// template <class T> T* undeclare_reachable(T* p);

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int* p = new int;
    std::declare_reachable(p);
    assert(std::undeclare_reachable(p) == p);
    delete p;

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test get_new_handler

// XFAIL: LIBCXX-WINDOWS-FIXME

#include <new>
#include <cassert>

#include "test_macros.h"

void f1() {}
void f2() {}

int main(int, char**)
{
    assert(std::get_new_handler() == 0);
    std::set_new_handler(f1);
    assert(std::get_new_handler() == f1);
    std::set_new_handler(f2);
    assert(std::get_new_handler() == f2);

  return 0;
}

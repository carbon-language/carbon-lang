//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr(nullptr_t)

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    std::shared_ptr<int> p(nullptr);
    assert(p.use_count() == 0);
    assert(p.get() == 0);
  }

  {
    std::shared_ptr<int const> p(nullptr);
    assert(p.use_count() == 0);
    assert(p.get() == 0);
  }

  return 0;
}

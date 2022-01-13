//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// pointer_safety get_pointer_safety();

// UNSUPPORTED: c++03

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    static_assert(std::is_enum<std::pointer_safety>::value, "");
    static_assert(!std::is_convertible<std::pointer_safety, int>::value, "");
    static_assert(std::is_same<
        std::underlying_type<std::pointer_safety>::type,
        unsigned char
    >::value, "");
  }
  {
    std::pointer_safety r = std::get_pointer_safety();
    assert(r == std::pointer_safety::relaxed ||
           r == std::pointer_safety::preferred ||
           r == std::pointer_safety::strict);
  }
  // Regression test for https://llvm.org/PR26961
  {
    std::pointer_safety d;
    d = std::get_pointer_safety();
    assert(d == std::get_pointer_safety());
  }

  return 0;
}

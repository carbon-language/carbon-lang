//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
// UNSUPPORTED: windows

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
// UNSUPPORTED: libcxx-no-debug-mode

// test array<T, 0>::operator[] raises a debug error.

#include <array>
#include "test_macros.h"
#include "debug_mode_helper.h"

int main(int, char**)
{
  {
    typedef std::array<int, 0> C;
    C c = {};
    C const& cc = c;
    EXPECT_DEATH( c[0] );
    EXPECT_DEATH( c[1] );
    EXPECT_DEATH( cc[0] );
    EXPECT_DEATH( cc[1] );
  }
  {
    typedef std::array<const int, 0> C;
    C c = {{}};
    C const& cc = c;
    EXPECT_DEATH( c[0] );
    EXPECT_DEATH( c[1] );
    EXPECT_DEATH( cc[0] );
    EXPECT_DEATH( cc[1] );
  }

  return 0;
}

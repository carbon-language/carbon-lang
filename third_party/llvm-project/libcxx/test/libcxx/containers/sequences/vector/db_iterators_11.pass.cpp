//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Index iterator out of bounds.

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "debug_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  typedef int T;
  typedef std::vector<T, min_allocator<T> > C;
  C c(1);
  C::iterator i = c.begin();
  assert(i[0] == 0);
  TEST_LIBCPP_ASSERT_FAILURE(i[1], "Attempted to subscript an iterator outside its valid range");

  return 0;
}

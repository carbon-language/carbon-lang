//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator first, const_iterator last); with second iterator from another container

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "test_macros.h"
#include "debug_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
  S l1("123");
  S l2("123");
  TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin(), l2.cbegin() + 1), "Attempted to compare incomparable iterators");

  return 0;
}

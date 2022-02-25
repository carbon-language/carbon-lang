// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_EXT and _LIBCPP_NODISCARD_AFTER_CXX17 are defined
// to the appropriate warning-generating attribute when _LIBCPP_ENABLE_NODISCARD
// is explicitly provided.

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_NODISCARD

#include <__config>

_LIBCPP_NODISCARD_EXT int foo() { return 42; }
_LIBCPP_NODISCARD_AFTER_CXX17 int bar() { return 42; }

int main(int, char**) {
  foo(); // expected-warning-re {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  bar(); // expected-warning-re {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  (void)foo(); // OK. void casts disable the diagnostic.
  (void)bar();

  return 0;
}

//===-- sanitizer_libc_test.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Tests for sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

// A regression test for internal_memmove() implementation.
TEST(SanitizerCommon, InternalMemmoveRegression) {
  char src[] = "Hello World";
  char *dest = src + 6;
  __sanitizer::internal_memmove(dest, src, 5);
  EXPECT_EQ(dest[0], src[0]);
  EXPECT_EQ(dest[4], src[4]);
}

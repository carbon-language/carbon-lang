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

TEST(SanitizerCommon, mem_is_zero) {
  size_t size = 128;
  char *x = new char[size];
  memset(x, 0, size);
  for (size_t pos = 0; pos < size; pos++) {
    x[pos] = 1;
    for (size_t beg = 0; beg < size; beg++) {
      for (size_t end = beg; end < size; end++) {
        // fprintf(stderr, "pos %zd beg %zd end %zd \n", pos, beg, end);
        if (beg <= pos && pos < end)
          EXPECT_FALSE(__sanitizer::mem_is_zero(x + beg, end - beg));
        else
          EXPECT_TRUE(__sanitizer::mem_is_zero(x + beg, end - beg));
      }
    }
    x[pos] = 0;
  }
  delete [] x;
}

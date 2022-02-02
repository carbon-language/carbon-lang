//===-- interception_linux_test.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for interception_linux.h.
//
//===----------------------------------------------------------------------===//

// Do not declare isdigit in ctype.h.
#define __NO_CTYPE

#include "interception/interception.h"

#include "gtest/gtest.h"

// Too slow for debug build
#if !SANITIZER_DEBUG
#if SANITIZER_LINUX

static int InterceptorFunctionCalled;

DECLARE_REAL(int, isdigit, int);

INTERCEPTOR(int, isdigit, int d) {
  ++InterceptorFunctionCalled;
  return d >= '0' && d <= '9';
}

namespace __interception {

TEST(Interception, InterceptFunction) {
  uptr malloc_address = 0;
  EXPECT_TRUE(InterceptFunction("malloc", &malloc_address, 0, 0));
  EXPECT_NE(0U, malloc_address);
  EXPECT_FALSE(InterceptFunction("malloc", &malloc_address, 0, 1));

  uptr dummy_address = 0;
  EXPECT_FALSE(InterceptFunction("dummy_doesnt_exist__", &dummy_address, 0, 0));
  EXPECT_EQ(0U, dummy_address);
}

TEST(Interception, Basic) {
  EXPECT_TRUE(INTERCEPT_FUNCTION(isdigit));

  // After interception, the counter should be incremented.
  InterceptorFunctionCalled = 0;
  EXPECT_NE(0, isdigit('1'));
  EXPECT_EQ(1, InterceptorFunctionCalled);
  EXPECT_EQ(0, isdigit('a'));
  EXPECT_EQ(2, InterceptorFunctionCalled);

  // Calling the REAL function should not affect the counter.
  InterceptorFunctionCalled = 0;
  EXPECT_NE(0, REAL(isdigit)('1'));
  EXPECT_EQ(0, REAL(isdigit)('a'));
  EXPECT_EQ(0, InterceptorFunctionCalled);
}

}  // namespace __interception

#endif  // SANITIZER_LINUX
#endif  // #if !SANITIZER_DEBUG

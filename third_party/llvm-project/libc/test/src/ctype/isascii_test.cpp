//===-- Unittests for isascii----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/UnitTest/Test.h"

#include "src/ctype/isascii.h"

TEST(LlvmLibcIsAscii, DefaultLocale) {
  // Loops through all characters, verifying that ascii characters
  //    (which are all 7 bit unsigned integers)
  // return a non-zero integer and everything else returns zero.
  for (int ch = 0; ch < 255; ++ch) {
    if (ch <= 0x7f)
      EXPECT_NE(__llvm_libc::isascii(ch), 0);
    else
      EXPECT_EQ(__llvm_libc::isascii(ch), 0);
  }
}

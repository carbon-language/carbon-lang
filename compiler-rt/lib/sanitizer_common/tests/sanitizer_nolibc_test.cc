//===-- sanitizer_nolibc_test.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for libc independence of sanitizer_common.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#include "gtest/gtest.h"

#include <stdlib.h>

extern const char *argv0;

#if SANITIZER_LINUX && defined(__x86_64__)
TEST(SanitizerCommon, NolibcMain) {
  std::string NolibcTestPath = argv0;
  NolibcTestPath += "-Nolibc";
  int status = system(NolibcTestPath.c_str());
  EXPECT_EQ(true, WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}
#endif

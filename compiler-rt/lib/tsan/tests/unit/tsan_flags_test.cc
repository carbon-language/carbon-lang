//===-- tsan_flags_test.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_flags.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"

namespace __tsan {

TEST(Flags, Basic) {
  ScopedInRtl in_rtl;
  // At least should not crash.
  Flags f;
  InitializeFlags(&f, 0);
  InitializeFlags(&f, "");
}

TEST(Flags, DefaultValues) {
  ScopedInRtl in_rtl;
  Flags f;

  f.enable_annotations = false;
  f.exitcode = -11;
  InitializeFlags(&f, "");
  EXPECT_EQ(66, f.exitcode);
  EXPECT_EQ(true, f.enable_annotations);
}

}  // namespace __tsan

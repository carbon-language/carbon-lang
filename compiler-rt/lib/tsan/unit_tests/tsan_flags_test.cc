//===-- tsan_flags_test.cc --------------------------------------*- C++ -*-===//
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
  Flags f = {};
  InitializeFlags(&f, 0);
  InitializeFlags(&f, "");
}

TEST(Flags, ParseBool) {
  ScopedInRtl in_rtl;
  Flags f = {};

  f.enable_annotations = false;
  InitializeFlags(&f, "enable_annotations");
  EXPECT_EQ(f.enable_annotations, true);

  f.enable_annotations = false;
  InitializeFlags(&f, "--enable_annotations");
  EXPECT_EQ(f.enable_annotations, true);

  f.enable_annotations = false;
  InitializeFlags(&f, "--enable_annotations=1");
  EXPECT_EQ(f.enable_annotations, true);

  // This flag is false by default.
  f.force_seq_cst_atomics = false;
  InitializeFlags(&f, "--force_seq_cst_atomics=1");
  EXPECT_EQ(f.force_seq_cst_atomics, true);

  f.enable_annotations = true;
  InitializeFlags(&f, "asdas enable_annotations=0 asdasd");
  EXPECT_EQ(f.enable_annotations, false);

  f.enable_annotations = true;
  InitializeFlags(&f, "   --enable_annotations=0   ");
  EXPECT_EQ(f.enable_annotations, false);
}

TEST(Flags, ParseInt) {
  ScopedInRtl in_rtl;
  Flags f = {};

  f.exitcode = -11;
  InitializeFlags(&f, "exitcode");
  EXPECT_EQ(f.exitcode, 0);

  f.exitcode = -11;
  InitializeFlags(&f, "--exitcode=");
  EXPECT_EQ(f.exitcode, 0);

  f.exitcode = -11;
  InitializeFlags(&f, "--exitcode=42");
  EXPECT_EQ(f.exitcode, 42);

  f.exitcode = -11;
  InitializeFlags(&f, "--exitcode=-42");
  EXPECT_EQ(f.exitcode, -42);
}

TEST(Flags, ParseStr) {
  ScopedInRtl in_rtl;
  Flags f = {};

  InitializeFlags(&f, 0);
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, ""));
  FinalizeFlags(&f);

  InitializeFlags(&f, "strip_path_prefix");
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, ""));
  FinalizeFlags(&f);

  InitializeFlags(&f, "--strip_path_prefix=");
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, ""));
  FinalizeFlags(&f);

  InitializeFlags(&f, "--strip_path_prefix=abc");
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, "abc"));
  FinalizeFlags(&f);

  InitializeFlags(&f, "--strip_path_prefix='abc zxc'");
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, "abc zxc"));
  FinalizeFlags(&f);

  InitializeFlags(&f, "--strip_path_prefix=\"abc zxc\"");
  EXPECT_EQ(0, strcmp(f.strip_path_prefix, "abc zxc"));
  FinalizeFlags(&f);
}

}  // namespace __tsan

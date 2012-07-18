//===-- sanitizer_flags_test.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "gtest/gtest.h"

#include "tsan_rtl.h"  // FIXME: break dependency from TSan runtime.
using __tsan::ScopedInRtl;

#include <string.h>

namespace __sanitizer {

static const char kFlagName[] = "flag_name";

template <typename T>
static void TestFlag(T start_value, const char *env, T final_value) {
  T flag = start_value;
  ParseFlag(env, &flag, kFlagName);
  EXPECT_EQ(final_value, flag);
}

static void TestStrFlag(const char *start_value, const char *env,
                        const char *final_value) {
  const char *flag = start_value;
  ParseFlag(env, &flag, kFlagName);
  EXPECT_STREQ(final_value, flag);
}

TEST(SanitizerCommon, BooleanFlags) {
  ScopedInRtl in_rtl;
  TestFlag(true, "--flag_name", true);
  TestFlag(false, "flag_name", false);
  TestFlag(false, "--flag_name=1", true);
  TestFlag(true, "asdas flag_name=0 asdas", false);
  TestFlag(true, "    --flag_name=0   ", false);
  TestFlag(false, "flag_name=yes", true);
  TestFlag(false, "flag_name=true", true);
  TestFlag(true, "flag_name=no", false);
  TestFlag(true, "flag_name=false", false);
}

TEST(SanitizerCommon, IntFlags) {
  ScopedInRtl in_rtl;
  TestFlag(-11, 0, -11);
  TestFlag(-11, "flag_name", 0);
  TestFlag(-11, "--flag_name=", 0);
  TestFlag(-11, "--flag_name=42", 42);
  TestFlag(-11, "--flag_name=-42", -42);
}

TEST(SanitizerCommon, StrFlags) {
  ScopedInRtl in_rtl;
  TestStrFlag("zzz", 0, "zzz");
  TestStrFlag("zzz", "flag_name", "");
  TestStrFlag("zzz", "--flag_name=", "");
  TestStrFlag("", "--flag_name=abc", "abc");
  TestStrFlag("", "--flag_name='abc zxc'", "abc zxc");
  TestStrFlag("", "--flag_name=\"abc qwe\" asd", "abc qwe");
}

}  // namespace __sanitizer

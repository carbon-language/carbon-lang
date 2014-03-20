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
#include "sanitizer_common/sanitizer_libc.h"
#include "gtest/gtest.h"

#include <string.h>

namespace __sanitizer {

static const char kFlagName[] = "flag_name";

template <typename T>
static void TestFlag(T start_value, const char *env, T final_value) {
  T flag = start_value;
  ParseFlag(env, &flag, kFlagName, "flag description");
  EXPECT_EQ(final_value, flag);
}

static void TestStrFlag(const char *start_value, const char *env,
                        const char *final_value) {
  const char *flag = start_value;
  ParseFlag(env, &flag, kFlagName, "flag description");
  EXPECT_EQ(0, internal_strcmp(final_value, flag));
}

TEST(SanitizerCommon, BooleanFlags) {
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
  TestFlag(-11, 0, -11);
  TestFlag(-11, "flag_name", 0);
  TestFlag(-11, "--flag_name=", 0);
  TestFlag(-11, "--flag_name=42", 42);
  TestFlag(-11, "--flag_name=-42", -42);
}

TEST(SanitizerCommon, StrFlags) {
  TestStrFlag("zzz", 0, "zzz");
  TestStrFlag("zzz", "flag_name", "");
  TestStrFlag("zzz", "--flag_name=", "");
  TestStrFlag("", "--flag_name=abc", "abc");
  TestStrFlag("", "--flag_name='abc zxc'", "abc zxc");
  TestStrFlag("", "--flag_name='abc zxcc'", "abc zxcc");
  TestStrFlag("", "--flag_name=\"abc qwe\" asd", "abc qwe");
  TestStrFlag("", "other_flag_name=zzz", "");
}

static void TestTwoFlags(const char *env, bool expected_flag1,
                         const char *expected_flag2) {
  bool flag1 = !expected_flag1;
  const char *flag2 = "";
  ParseFlag(env, &flag1, "flag1", "flag1 description");
  ParseFlag(env, &flag2, "flag2", "flag2 description");
  EXPECT_EQ(expected_flag1, flag1);
  EXPECT_EQ(0, internal_strcmp(flag2, expected_flag2));
}

TEST(SanitizerCommon, MultipleFlags) {
  TestTwoFlags("flag1=1 flag2='zzz'", true, "zzz");
  TestTwoFlags("flag2='qxx' flag1=0", false, "qxx");
  TestTwoFlags("flag1=false:flag2='zzz'", false, "zzz");
  TestTwoFlags("flag2=qxx:flag1=yes", true, "qxx");
}

}  // namespace __sanitizer

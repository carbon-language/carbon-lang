//===-- tsan_printf_test.cc -----------------------------------------------===//
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
#include "tsan_rtl.h"
#include "gtest/gtest.h"

#include <string.h>
#include <limits.h>

namespace __tsan {

TEST(Printf, Basic) {
  char buf[1024];
  uptr len = Snprintf(buf, sizeof(buf),
      "a%db%ldc%lldd%ue%luf%llug%xh%lxq%llxw%pe%sr",
      (int)-1, (long)-2, (long long)-3,  // NOLINT
      (unsigned)-4, (unsigned long)5, (unsigned long long)6,  // NOLINT
      (unsigned)10, (unsigned long)11, (unsigned long long)12,  // NOLINT
      (void*)0x123, "_string_");
  EXPECT_EQ(len, strlen(buf));
  EXPECT_EQ(0, strcmp(buf, "a-1b-2c-3d4294967292e5f6gahbqcw"
                           "0x000000000123e_string_r"));
}

TEST(Printf, OverflowStr) {
  char buf[] = "123456789";
  uptr len = Snprintf(buf, 4, "%s", "abcdef");
  EXPECT_EQ(len, (uptr)6);
  EXPECT_EQ(0, strcmp(buf, "abc"));
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowInt) {
  char buf[] = "123456789";
  Snprintf(buf, 4, "%d", -123456789);
  EXPECT_EQ(0, strcmp(buf, "-12"));
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowUint) {
  char buf[] = "123456789";
  Snprintf(buf, 4, "a%llx", (long long)0x123456789);  // NOLINT
  EXPECT_EQ(0, strcmp(buf, "a12"));
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

TEST(Printf, OverflowPtr) {
  char buf[] = "123456789";
  Snprintf(buf, 4, "%p", (void*)0x123456789);
  EXPECT_EQ(0, strcmp(buf, "0x0"));
  EXPECT_EQ(buf[3], 0);
  EXPECT_EQ(buf[4], '5');
  EXPECT_EQ(buf[5], '6');
  EXPECT_EQ(buf[6], '7');
  EXPECT_EQ(buf[7], '8');
  EXPECT_EQ(buf[8], '9');
  EXPECT_EQ(buf[9], 0);
}

template<typename T>
static void TestMinMax(const char *fmt, T min, T max) {
  char buf[1024];
  uptr len = Snprintf(buf, sizeof(buf), fmt, min, max);
  char buf2[1024];
  snprintf(buf2, sizeof(buf2), fmt, min, max);
  EXPECT_EQ(len, strlen(buf));
  EXPECT_EQ(0, strcmp(buf, buf2));
}

TEST(Printf, MinMax) {
  TestMinMax<int>("%d-%d", INT_MIN, INT_MAX);  // NOLINT
  TestMinMax<long>("%ld-%ld", LONG_MIN, LONG_MAX);  // NOLINT
  TestMinMax<long long>("%lld-%lld", LLONG_MIN, LLONG_MAX);  // NOLINT
  TestMinMax<unsigned>("%u-%u", 0, UINT_MAX);  // NOLINT
  TestMinMax<unsigned long>("%lu-%lu", 0, ULONG_MAX);  // NOLINT
  TestMinMax<unsigned long long>("%llu-%llu", 0, ULLONG_MAX);  // NOLINT
  TestMinMax<unsigned>("%x-%x", 0, UINT_MAX);  // NOLINT
  TestMinMax<unsigned long>("%lx-%lx", 0, ULONG_MAX);  // NOLINT
  TestMinMax<unsigned long long>("%llx-%llx", 0, ULLONG_MAX);  // NOLINT
}

}  // namespace __tsan

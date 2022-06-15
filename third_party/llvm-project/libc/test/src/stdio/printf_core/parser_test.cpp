//===-- Unittests for the printf Parser -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Bit.h"
#include "src/__support/arg_list.h"
#include "src/stdio/printf_core/parser.h"

#include <stdarg.h>

#include "utils/UnitTest/PrintfMatcher.h"
#include "utils/UnitTest/Test.h"

void init(const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::printf_core::Parser parser(str, v);
}

void evaluate(__llvm_libc::printf_core::FormatSection *format_arr,
              const char *__restrict str, ...) {
  va_list vlist;
  va_start(vlist, str);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  __llvm_libc::printf_core::Parser parser(str, v);

  for (auto cur_section = parser.get_next_section(); cur_section.raw_len > 0;
       cur_section = parser.get_next_section()) {
    *format_arr = cur_section;
    ++format_arr;
  }
}

TEST(LlvmLibcPrintfParserTest, Constructor) { init("test", 1, 2); }

TEST(LlvmLibcPrintfParserTest, EvalRaw) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "test";
  evaluate(format_arr, str);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = false;
  expected.raw_len = 4;
  expected.raw_string = str;

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
  // TODO: add checks that the format_arr after the last one has length 0
}

TEST(LlvmLibcPrintfParserTest, EvalSimple) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "test %% test";
  evaluate(format_arr, str);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = false;
  expected0.raw_len = 5;
  expected0.raw_string = str;

  ASSERT_FORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 2;
  expected1.raw_string = str + 5;
  expected1.conv_name = '%';

  ASSERT_FORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;
  expected2.raw_len = 5;
  expected2.raw_string = str + 7;

  ASSERT_FORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArg) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 2;
  expected.raw_string = str;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithFlags) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%+-0 #d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 7;
  expected.raw_string = str;
  expected.flags = static_cast<__llvm_libc::printf_core::FormatFlags>(
      __llvm_libc::printf_core::FormatFlags::FORCE_SIGN |
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED |
      __llvm_libc::printf_core::FormatFlags::LEADING_ZEROES |
      __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX |
      __llvm_libc::printf_core::FormatFlags::ALTERNATE_FORM);
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithWidth) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%12d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 4;
  expected.raw_string = str;
  expected.min_width = 12;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithPrecision) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%.34d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 5;
  expected.raw_string = str;
  expected.precision = 34;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithTrivialPrecision) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%.d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 3;
  expected.raw_string = str;
  expected.precision = 0;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithShortLengthModifier) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%hd";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 3;
  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::h;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithLongLengthModifier) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%lld";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 4;
  expected.raw_string = str;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::ll;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalOneArgWithAllOptions) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "% -056.78jd";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 11;
  expected.raw_string = str;
  expected.flags = static_cast<__llvm_libc::printf_core::FormatFlags>(
      __llvm_libc::printf_core::FormatFlags::LEFT_JUSTIFIED |
      __llvm_libc::printf_core::FormatFlags::LEADING_ZEROES |
      __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX);
  expected.min_width = 56;
  expected.precision = 78;
  expected.length_modifier = __llvm_libc::printf_core::LengthModifier::j;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, EvalThreeArgs) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%d%f%s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg1, arg2, arg3);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;
  expected0.raw_len = 2;
  expected0.raw_string = str;
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 2;
  expected1.raw_string = str + 2;
  expected1.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_FORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;
  expected2.raw_len = 2;
  expected2.raw_string = str + 4;
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_FORMAT_EQ(expected2, format_arr[2]);
}

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

TEST(LlvmLibcPrintfParserTest, IndexModeOneArg) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%1$d";
  int arg1 = 12345;
  evaluate(format_arr, str, arg1);

  __llvm_libc::printf_core::FormatSection expected;
  expected.has_conv = true;
  expected.raw_len = 4;
  expected.raw_string = str;
  expected.conv_val_raw = arg1;
  expected.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected, format_arr[0]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeThreeArgsSequential) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%1$d%2$f%3$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg1, arg2, arg3);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;
  expected0.raw_len = 4;
  expected0.raw_string = str;
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 4;
  expected1.raw_string = str + 4;
  expected1.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_FORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;
  expected2.raw_len = 4;
  expected2.raw_string = str + 8;
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_FORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeThreeArgsReverse) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%3$d%2$f%1$s";
  int arg1 = 12345;
  double arg2 = 123.45;
  const char *arg3 = "12345";
  evaluate(format_arr, str, arg3, arg2, arg1);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2;
  expected0.has_conv = true;
  expected0.raw_len = 4;
  expected0.raw_string = str;
  expected0.conv_val_raw = arg1;
  expected0.conv_name = 'd';

  ASSERT_FORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 4;
  expected1.raw_string = str + 4;
  expected1.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected1.conv_name = 'f';

  ASSERT_FORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = true;
  expected2.raw_len = 4;
  expected2.raw_string = str + 8;
  expected2.conv_val_ptr = const_cast<char *>(arg3);
  expected2.conv_name = 's';

  ASSERT_FORMAT_EQ(expected2, format_arr[2]);
}

TEST(LlvmLibcPrintfParserTest, IndexModeTenArgsRandom) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "%6$d%3$d%7$d%2$d%8$d%1$d%4$d%9$d%5$d%10$d";
  int args[10] = {6, 4, 2, 7, 9, 1, 3, 5, 8, 10};
  evaluate(format_arr, str, args[0], args[1], args[2], args[3], args[4],
           args[5], args[6], args[7], args[8], args[9]);

  for (size_t i = 0; i < 10; ++i) {
    __llvm_libc::printf_core::FormatSection expected;
    expected.has_conv = true;
    expected.raw_len = 4 + (i >= 9 ? 1 : 0);
    expected.raw_string = str + (4 * i);
    expected.conv_val_raw = i + 1;
    expected.conv_name = 'd';
    EXPECT_FORMAT_EQ(expected, format_arr[i]);
  }
}

TEST(LlvmLibcPrintfParserTest, IndexModeComplexParsing) {
  __llvm_libc::printf_core::FormatSection format_arr[10];
  const char *str = "normal text %3$llu %% %2$ *4$f %2$ .*4$f %1$1.1c";
  char arg1 = '1';
  double arg2 = 123.45;
  unsigned long long arg3 = 12345;
  int arg4 = 10;
  evaluate(format_arr, str, arg1, arg2, arg3, arg4);

  __llvm_libc::printf_core::FormatSection expected0, expected1, expected2,
      expected3, expected4, expected5, expected6, expected7, expected8,
      expected9;

  expected0.has_conv = false;
  expected0.raw_len = 12;
  expected0.raw_string = str;

  EXPECT_FORMAT_EQ(expected0, format_arr[0]);

  expected1.has_conv = true;
  expected1.raw_len = 6;
  expected1.raw_string = str + 12;
  expected1.length_modifier = __llvm_libc::printf_core::LengthModifier::ll;
  expected1.conv_val_raw = arg3;
  expected1.conv_name = 'u';

  EXPECT_FORMAT_EQ(expected1, format_arr[1]);

  expected2.has_conv = false;
  expected2.raw_len = 1;
  expected2.raw_string = str + 18;

  EXPECT_FORMAT_EQ(expected2, format_arr[2]);

  expected3.has_conv = true;
  expected3.raw_len = 2;
  expected3.raw_string = str + 19;
  expected3.conv_name = '%';

  EXPECT_FORMAT_EQ(expected3, format_arr[3]);

  expected4.has_conv = false;
  expected4.raw_len = 1;
  expected4.raw_string = str + 21;

  EXPECT_FORMAT_EQ(expected4, format_arr[4]);

  expected5.has_conv = true;
  expected5.raw_len = 8;
  expected5.raw_string = str + 22;
  expected5.flags = __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX;
  expected5.min_width = arg4;
  expected5.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected5.conv_name = 'f';

  EXPECT_FORMAT_EQ(expected5, format_arr[5]);

  expected6.has_conv = false;
  expected6.raw_len = 1;
  expected6.raw_string = str + 30;

  EXPECT_FORMAT_EQ(expected6, format_arr[6]);

  expected7.has_conv = true;
  expected7.raw_len = 9;
  expected7.raw_string = str + 31;
  expected7.flags = __llvm_libc::printf_core::FormatFlags::SPACE_PREFIX;
  expected7.precision = arg4;
  expected7.conv_val_raw = __llvm_libc::bit_cast<uint64_t>(arg2);
  expected7.conv_name = 'f';

  EXPECT_FORMAT_EQ(expected7, format_arr[7]);

  expected8.has_conv = false;
  expected8.raw_len = 1;
  expected8.raw_string = str + 40;

  EXPECT_FORMAT_EQ(expected8, format_arr[8]);

  expected9.has_conv = true;
  expected9.raw_len = 7;
  expected9.raw_string = str + 41;
  expected9.min_width = 1;
  expected9.precision = 1;
  expected9.conv_val_raw = arg1;
  expected9.conv_name = 'c';

  EXPECT_FORMAT_EQ(expected9, format_arr[9]);
}

#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

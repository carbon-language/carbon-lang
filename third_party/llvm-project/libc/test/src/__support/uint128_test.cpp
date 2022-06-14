//===-- Unittests for the 128 bit integer class ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/UInt.h"

#include "utils/UnitTest/Test.h"

using UInt128 = __llvm_libc::cpp::UInt<128>;

TEST(LlvmLibcUInt128ClassTest, BasicInit) {
  UInt128 empty;
  UInt128 half_val(12345);
  UInt128 full_val({12345, 67890});
  ASSERT_TRUE(half_val != full_val);
}

TEST(LlvmLibcUInt128ClassTest, AdditionTests) {
  UInt128 val1(12345);
  UInt128 val2(54321);
  UInt128 result1(66666);
  EXPECT_EQ(val1 + val2, result1);
  EXPECT_EQ((val1 + val2), (val2 + val1)); // addition is reciprocal

  // Test overflow
  UInt128 val3({0xf000000000000001, 0});
  UInt128 val4({0x100000000000000f, 0});
  UInt128 result2({0x10, 0x1});
  EXPECT_EQ(val3 + val4, result2);
  EXPECT_EQ(val3 + val4, val4 + val3);
}

TEST(LlvmLibcUInt128ClassTest, MultiplicationTests) {
  UInt128 val1({5, 0});
  UInt128 val2({10, 0});
  UInt128 result1({50, 0});
  EXPECT_EQ((val1 * val2), result1);
  EXPECT_EQ((val1 * val2), (val2 * val1)); // multiplication is reciprocal

  // Check that the multiplication works accross the whole number
  UInt128 val3({0xf, 0});
  UInt128 val4({0x1111111111111111, 0x1111111111111111});
  UInt128 result2({0xffffffffffffffff, 0xffffffffffffffff});
  EXPECT_EQ((val3 * val4), result2);
  EXPECT_EQ((val3 * val4), (val4 * val3));

  // Check that multiplication doesn't reorder the bits.
  UInt128 val5({2, 0});
  UInt128 val6({0x1357024675316420, 0x0123456776543210});
  UInt128 result3({0x26ae048cea62c840, 0x02468aceeca86420});

  EXPECT_EQ((val5 * val6), result3);
  EXPECT_EQ((val5 * val6), (val6 * val5));

  // Make sure that multiplication handles overflow correctly.
  UInt128 val7(2);
  UInt128 val8({0x8000800080008000, 0x8000800080008000});
  UInt128 result4({0x0001000100010000, 0x0001000100010001});
  EXPECT_EQ((val7 * val8), result4);
  EXPECT_EQ((val7 * val8), (val8 * val7));

  // val9 is the 128 bit mantissa of 1e60 as a float, val10 is the mantissa for
  // 1e-60. They almost cancel on the high bits, but the result we're looking
  // for is just the low bits. The full result would be
  // 0x7fffffffffffffffffffffffffffffff3a4f32d17f40d08f917cf11d1e039c50
  UInt128 val9({0x01D762422C946590, 0x9F4F2726179A2245});
  UInt128 val10({0x3792F412CB06794D, 0xCDB02555653131B6});
  UInt128 result5({0x917cf11d1e039c50, 0x3a4f32d17f40d08f});
  EXPECT_EQ((val9 * val10), result5);
  EXPECT_EQ((val9 * val10), (val10 * val9));
}

TEST(LlvmLibcUInt128ClassTest, ShiftLeftTests) {
  UInt128 val1(0x0123456789abcdef);
  UInt128 result1(0x123456789abcdef0);
  EXPECT_EQ((val1 << 4), result1);

  UInt128 val2({0x13579bdf02468ace, 0x123456789abcdef0});
  UInt128 result2({0x02468ace00000000, 0x9abcdef013579bdf});
  EXPECT_EQ((val2 << 32), result2);

  UInt128 result3({0, 0x13579bdf02468ace});
  EXPECT_EQ((val2 << 64), result3);

  UInt128 result4({0, 0x02468ace00000000});
  EXPECT_EQ((val2 << 96), result4);

  UInt128 result5({0, 0x2468ace000000000});
  EXPECT_EQ((val2 << 100), result5);

  UInt128 result6({0, 0});
  EXPECT_EQ((val2 << 128), result6);
  EXPECT_EQ((val2 << 256), result6);
}

TEST(LlvmLibcUInt128ClassTest, ShiftRightTests) {
  UInt128 val1(0x0123456789abcdef);
  UInt128 result1(0x00123456789abcde);
  EXPECT_EQ((val1 >> 4), result1);

  UInt128 val2({0x13579bdf02468ace, 0x123456789abcdef0});
  UInt128 result2({0x9abcdef013579bdf, 0x0000000012345678});
  EXPECT_EQ((val2 >> 32), result2);

  UInt128 result3({0x123456789abcdef0, 0});
  EXPECT_EQ((val2 >> 64), result3);

  UInt128 result4({0x0000000012345678, 0});
  EXPECT_EQ((val2 >> 96), result4);

  UInt128 result5({0x0000000001234567, 0});
  EXPECT_EQ((val2 >> 100), result5);

  UInt128 result6({0, 0});
  EXPECT_EQ((val2 >> 128), result6);
  EXPECT_EQ((val2 >> 256), result6);
}

TEST(LlvmLibcUInt128ClassTest, AndTests) {
  UInt128 base({0xffff00000000ffff, 0xffffffff00000000});
  UInt128 val128({0xf0f0f0f00f0f0f0f, 0xff00ff0000ff00ff});
  uint64_t val64 = 0xf0f0f0f00f0f0f0f;
  int val32 = 0x0f0f0f0f;
  UInt128 result128({0xf0f0000000000f0f, 0xff00ff0000000000});
  UInt128 result64(0xf0f0000000000f0f);
  UInt128 result32(0x00000f0f);
  EXPECT_EQ((base & val128), result128);
  EXPECT_EQ((base & val64), result64);
  EXPECT_EQ((base & val32), result32);
}

TEST(LlvmLibcUInt128ClassTest, OrTests) {
  UInt128 base({0xffff00000000ffff, 0xffffffff00000000});
  UInt128 val128({0xf0f0f0f00f0f0f0f, 0xff00ff0000ff00ff});
  uint64_t val64 = 0xf0f0f0f00f0f0f0f;
  int val32 = 0x0f0f0f0f;
  UInt128 result128({0xfffff0f00f0fffff, 0xffffffff00ff00ff});
  UInt128 result64({0xfffff0f00f0fffff, 0xffffffff00000000});
  UInt128 result32({0xffff00000f0fffff, 0xffffffff00000000});
  EXPECT_EQ((base | val128), result128);
  EXPECT_EQ((base | val64), result64);
  EXPECT_EQ((base | val32), result32);
}

TEST(LlvmLibcUInt128ClassTest, EqualsTests) {
  UInt128 a1({0xffffffff00000000, 0xffff00000000ffff});
  UInt128 a2({0xffffffff00000000, 0xffff00000000ffff});
  UInt128 b({0xff00ff0000ff00ff, 0xf0f0f0f00f0f0f0f});
  UInt128 a_reversed({0xffff00000000ffff, 0xffffffff00000000});
  UInt128 a_upper(0xffff00000000ffff);
  UInt128 a_lower(0xffffffff00000000);
  ASSERT_TRUE(a1 == a1);
  ASSERT_TRUE(a1 == a2);
  ASSERT_FALSE(a1 == b);
  ASSERT_FALSE(a1 == a_reversed);
  ASSERT_FALSE(a1 == a_lower);
  ASSERT_FALSE(a1 == a_upper);
  ASSERT_TRUE(a_lower != a_upper);
}

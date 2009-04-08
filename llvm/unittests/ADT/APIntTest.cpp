//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;

namespace {

// Make the Google Test failure output equivalent to APInt::dump()
std::ostream& operator<<(std::ostream &OS, const llvm::APInt& I) {
  llvm::raw_os_ostream raw_os(OS);

  SmallString<40> S, U;
  I.toStringUnsigned(U);
  I.toStringSigned(S);
  raw_os << "APInt(" << I.getBitWidth()<< "b, "
         << U.c_str() << "u " << S.c_str() << "s)";
  raw_os.flush();
  return OS;
}

// Test that APInt shift left works when bitwidth > 64 and shiftamt == 0
TEST(APIntTest, ShiftLeftByZero) {
  APInt One = APInt::getNullValue(65) + 1;
  APInt Shl = One.shl(0);
  EXPECT_EQ(true, Shl[0]);
  EXPECT_EQ(false, Shl[1]);
}

TEST(APIntTest, i128_NegativeCount) {
  APInt Minus3(128, static_cast<uint64_t>(-3), true);
  EXPECT_EQ(126u, Minus3.countLeadingOnes());
  EXPECT_EQ(-3, Minus3.getSExtValue());

  APInt Minus1(128, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(0u, Minus1.countLeadingZeros());
  EXPECT_EQ(128u, Minus1.countLeadingOnes());
  EXPECT_EQ(128u, Minus1.getActiveBits());
  EXPECT_EQ(0u, Minus1.countTrailingZeros());
  EXPECT_EQ(128u, Minus1.countTrailingOnes());
  EXPECT_EQ(128u, Minus1.countPopulation());
  EXPECT_EQ(-1, Minus1.getSExtValue());
}

TEST(APIntTest, i33_Count) {
  APInt i33minus2(33, static_cast<uint64_t>(-2), true);
  EXPECT_EQ(0u, i33minus2.countLeadingZeros());
  EXPECT_EQ(32u, i33minus2.countLeadingOnes());
  EXPECT_EQ(33u, i33minus2.getActiveBits());
  EXPECT_EQ(1u, i33minus2.countTrailingZeros());
  EXPECT_EQ(32u, i33minus2.countPopulation());
  EXPECT_EQ(-2, i33minus2.getSExtValue());
  EXPECT_EQ(((uint64_t)-2)&((1ull<<33) -1), i33minus2.getZExtValue());
}

TEST(APIntTest, i65_Count) {
  APInt i65minus(65, 0, true);
  i65minus.set(64);
  EXPECT_EQ(0u, i65minus.countLeadingZeros());
  EXPECT_EQ(1u, i65minus.countLeadingOnes());
  EXPECT_EQ(65u, i65minus.getActiveBits());
  EXPECT_EQ(64u, i65minus.countTrailingZeros());
  EXPECT_EQ(1u, i65minus.countPopulation());
}

TEST(APIntTest, i128_PositiveCount) {
  APInt u128max = APInt::getAllOnesValue(128);
  EXPECT_EQ(128u, u128max.countLeadingOnes());
  EXPECT_EQ(0u, u128max.countLeadingZeros());
  EXPECT_EQ(128u, u128max.getActiveBits());
  EXPECT_EQ(0u, u128max.countTrailingZeros());
  EXPECT_EQ(128u, u128max.countTrailingOnes());
  EXPECT_EQ(128u, u128max.countPopulation());

  APInt u64max(128, static_cast<uint64_t>(-1), false);
  EXPECT_EQ(64u, u64max.countLeadingZeros());
  EXPECT_EQ(0u, u64max.countLeadingOnes());
  EXPECT_EQ(64u, u64max.getActiveBits());
  EXPECT_EQ(0u, u64max.countTrailingZeros());
  EXPECT_EQ(64u, u64max.countTrailingOnes());
  EXPECT_EQ(64u, u64max.countPopulation());
  EXPECT_EQ((uint64_t)~0ull, u64max.getZExtValue());

  APInt zero(128, 0, true);
  EXPECT_EQ(128u, zero.countLeadingZeros());
  EXPECT_EQ(0u, zero.countLeadingOnes());
  EXPECT_EQ(0u, zero.getActiveBits());
  EXPECT_EQ(128u, zero.countTrailingZeros());
  EXPECT_EQ(0u, zero.countTrailingOnes());
  EXPECT_EQ(0u, zero.countPopulation());
  EXPECT_EQ(0u, zero.getSExtValue());
  EXPECT_EQ(0u, zero.getZExtValue());

  APInt one(128, 1, true);
  EXPECT_EQ(127u, one.countLeadingZeros());
  EXPECT_EQ(0u, one.countLeadingOnes());
  EXPECT_EQ(1u, one.getActiveBits());
  EXPECT_EQ(0u, one.countTrailingZeros());
  EXPECT_EQ(1u, one.countTrailingOnes());
  EXPECT_EQ(1u, one.countPopulation());
  EXPECT_EQ(1, one.getSExtValue());
  EXPECT_EQ(1u, one.getZExtValue());
}

TEST(APIntTest, i1) {
  const APInt neg_two(1, static_cast<uint64_t>(-2), true);
  const APInt neg_one(1, static_cast<uint64_t>(-1), true);
  const APInt zero(1, 0);
  const APInt one(1, 1);
  const APInt two(1, 2);

  EXPECT_EQ(0, neg_two.getSExtValue());
  EXPECT_EQ(-1, neg_one.getSExtValue());
  EXPECT_EQ(1u, neg_one.getZExtValue());
  EXPECT_EQ(0u, zero.getZExtValue());
  EXPECT_EQ(-1, one.getSExtValue());
  EXPECT_EQ(1u, one.getZExtValue());
  EXPECT_EQ(0u, two.getZExtValue());
  EXPECT_EQ(0, two.getSExtValue());

  // Basic equalities for 1-bit values.
  EXPECT_EQ(zero, two);
  EXPECT_EQ(zero, neg_two);
  EXPECT_EQ(one, neg_one);
  EXPECT_EQ(two, neg_two);

  // Additions.
  EXPECT_EQ(two, one + one);
  EXPECT_EQ(zero, neg_one + one);
  EXPECT_EQ(neg_two, neg_one + neg_one);

  // Subtractions.
  EXPECT_EQ(neg_two, neg_one - one);
  EXPECT_EQ(two, one - neg_one);
  EXPECT_EQ(zero, one - one);

  // Shifts.
  EXPECT_EQ(zero, one << one);
  EXPECT_EQ(one, one << zero);
  EXPECT_EQ(zero, one.shl(1));
  EXPECT_EQ(one, one.shl(0));
  EXPECT_EQ(zero, one.lshr(1));
  EXPECT_EQ(zero, one.ashr(1));

  // Multiplies.
  EXPECT_EQ(neg_one, neg_one * one);
  EXPECT_EQ(neg_one, one * neg_one);
  EXPECT_EQ(one, neg_one * neg_one);
  EXPECT_EQ(one, one * one);

  // Divides.
  EXPECT_EQ(neg_one, one.sdiv(neg_one));
  EXPECT_EQ(neg_one, neg_one.sdiv(one));
  EXPECT_EQ(one, neg_one.sdiv(neg_one));
  EXPECT_EQ(one, one.sdiv(one));

  EXPECT_EQ(neg_one, one.udiv(neg_one));
  EXPECT_EQ(neg_one, neg_one.udiv(one));
  EXPECT_EQ(one, neg_one.udiv(neg_one));
  EXPECT_EQ(one, one.udiv(one));

  // Remainders.
  EXPECT_EQ(zero, neg_one.srem(one));
  EXPECT_EQ(zero, neg_one.urem(one));
  EXPECT_EQ(zero, one.srem(neg_one));
}

}

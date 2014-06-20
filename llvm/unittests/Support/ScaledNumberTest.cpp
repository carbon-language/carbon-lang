//===- llvm/unittest/Support/ScaledNumberTest.cpp - ScaledPair tests -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ScaledNumber.h"

#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::ScaledNumbers;

namespace {

template <class UIntT> struct ScaledPair {
  UIntT D;
  int S;
  ScaledPair(const std::pair<UIntT, int16_t> &F) : D(F.first), S(F.second) {}
  ScaledPair(UIntT D, int S) : D(D), S(S) {}

  bool operator==(const ScaledPair<UIntT> &X) const {
    return D == X.D && S == X.S;
  }
};
template <class UIntT>
bool operator==(const std::pair<UIntT, int16_t> &L,
                const ScaledPair<UIntT> &R) {
  return ScaledPair<UIntT>(L) == R;
}
template <class UIntT>
void PrintTo(const ScaledPair<UIntT> &F, ::std::ostream *os) {
  *os << F.D << "*2^" << F.S;
}

typedef ScaledPair<uint32_t> SP32;
typedef ScaledPair<uint64_t> SP64;

TEST(ScaledNumberHelpersTest, getRounded) {
  EXPECT_EQ(getRounded32(0, 0, false), SP32(0, 0));
  EXPECT_EQ(getRounded32(0, 0, true), SP32(1, 0));
  EXPECT_EQ(getRounded32(20, 21, true), SP32(21, 21));
  EXPECT_EQ(getRounded32(UINT32_MAX, 0, false), SP32(UINT32_MAX, 0));
  EXPECT_EQ(getRounded32(UINT32_MAX, 0, true), SP32(1 << 31, 1));

  EXPECT_EQ(getRounded64(0, 0, false), SP64(0, 0));
  EXPECT_EQ(getRounded64(0, 0, true), SP64(1, 0));
  EXPECT_EQ(getRounded64(20, 21, true), SP64(21, 21));
  EXPECT_EQ(getRounded64(UINT32_MAX, 0, false), SP64(UINT32_MAX, 0));
  EXPECT_EQ(getRounded64(UINT32_MAX, 0, true), SP64(UINT64_C(1) << 32, 0));
  EXPECT_EQ(getRounded64(UINT64_MAX, 0, false), SP64(UINT64_MAX, 0));
  EXPECT_EQ(getRounded64(UINT64_MAX, 0, true), SP64(UINT64_C(1) << 63, 1));
}

TEST(FloatsTest, getAdjusted) {
  const uint64_t Max32In64 = UINT32_MAX;
  EXPECT_EQ(getAdjusted32(0), SP32(0, 0));
  EXPECT_EQ(getAdjusted32(0, 5), SP32(0, 5));
  EXPECT_EQ(getAdjusted32(UINT32_MAX), SP32(UINT32_MAX, 0));
  EXPECT_EQ(getAdjusted32(Max32In64 << 1), SP32(UINT32_MAX, 1));
  EXPECT_EQ(getAdjusted32(Max32In64 << 1, 1), SP32(UINT32_MAX, 2));
  EXPECT_EQ(getAdjusted32(Max32In64 << 31), SP32(UINT32_MAX, 31));
  EXPECT_EQ(getAdjusted32(Max32In64 << 32), SP32(UINT32_MAX, 32));
  EXPECT_EQ(getAdjusted32(Max32In64 + 1), SP32(1u << 31, 1));
  EXPECT_EQ(getAdjusted32(UINT64_MAX), SP32(1u << 31, 33));

  EXPECT_EQ(getAdjusted64(0), SP64(0, 0));
  EXPECT_EQ(getAdjusted64(0, 5), SP64(0, 5));
  EXPECT_EQ(getAdjusted64(UINT32_MAX), SP64(UINT32_MAX, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 1), SP64(Max32In64 << 1, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 1, 1), SP64(Max32In64 << 1, 1));
  EXPECT_EQ(getAdjusted64(Max32In64 << 31), SP64(Max32In64 << 31, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 << 32), SP64(Max32In64 << 32, 0));
  EXPECT_EQ(getAdjusted64(Max32In64 + 1), SP64(Max32In64 + 1, 0));
  EXPECT_EQ(getAdjusted64(UINT64_MAX), SP64(UINT64_MAX, 0));
}

TEST(PositiveFloatTest, getProduct) {
  // Zero.
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 0));
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 1));
  EXPECT_EQ(SP32(0, 0), getProduct32(0, 33));

  // Basic.
  EXPECT_EQ(SP32(6, 0), getProduct32(2, 3));
  EXPECT_EQ(SP32(UINT16_MAX / 3 * UINT16_MAX / 5 * 2, 0),
            getProduct32(UINT16_MAX / 3, UINT16_MAX / 5 * 2));

  // Overflow, no loss of precision.
  // ==> 0xf00010 * 0x1001
  // ==> 0xf00f00000 + 0x10010
  // ==> 0xf00f10010
  // ==> 0xf00f1001 * 2^4
  EXPECT_EQ(SP32(0xf00f1001, 4), getProduct32(0xf00010, 0x1001));

  // Overflow, loss of precision, rounds down.
  // ==> 0xf000070 * 0x1001
  // ==> 0xf00f000000 + 0x70070
  // ==> 0xf00f070070
  // ==> 0xf00f0700 * 2^8
  EXPECT_EQ(SP32(0xf00f0700, 8), getProduct32(0xf000070, 0x1001));

  // Overflow, loss of precision, rounds up.
  // ==> 0xf000080 * 0x1001
  // ==> 0xf00f000000 + 0x80080
  // ==> 0xf00f080080
  // ==> 0xf00f0801 * 2^8
  EXPECT_EQ(SP32(0xf00f0801, 8), getProduct32(0xf000080, 0x1001));

  // Reverse operand order.
  EXPECT_EQ(SP32(0, 0), getProduct32(1, 0));
  EXPECT_EQ(SP32(0, 0), getProduct32(33, 0));
  EXPECT_EQ(SP32(6, 0), getProduct32(3, 2));
  EXPECT_EQ(SP32(UINT16_MAX / 3 * UINT16_MAX / 5 * 2, 0),
            getProduct32(UINT16_MAX / 5 * 2, UINT16_MAX / 3));
  EXPECT_EQ(SP32(0xf00f1001, 4), getProduct32(0x1001, 0xf00010));
  EXPECT_EQ(SP32(0xf00f0700, 8), getProduct32(0x1001, 0xf000070));
  EXPECT_EQ(SP32(0xf00f0801, 8), getProduct32(0x1001, 0xf000080));

  // Round to overflow.
  EXPECT_EQ(SP64(UINT64_C(1) << 63, 64),
            getProduct64(UINT64_C(10376293541461622786),
                         UINT64_C(16397105843297379211)));

  // Big number with rounding.
  EXPECT_EQ(SP64(UINT64_C(9223372036854775810), 64),
            getProduct64(UINT64_C(18446744073709551556),
                         UINT64_C(9223372036854775840)));
}

TEST(PositiveFloatTest, Divide) {
  // Zero.
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 0));
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 1));
  EXPECT_EQ(SP32(0, 0), getQuotient32(0, 73));
  EXPECT_EQ(SP32(UINT32_MAX, INT16_MAX), getQuotient32(1, 0));
  EXPECT_EQ(SP32(UINT32_MAX, INT16_MAX), getQuotient32(6, 0));

  // Powers of two.
  EXPECT_EQ(SP32(1u << 31, -31), getQuotient32(1, 1));
  EXPECT_EQ(SP32(1u << 31, -30), getQuotient32(2, 1));
  EXPECT_EQ(SP32(1u << 31, -33), getQuotient32(4, 16));
  EXPECT_EQ(SP32(7u << 29, -29), getQuotient32(7, 1));
  EXPECT_EQ(SP32(7u << 29, -30), getQuotient32(7, 2));
  EXPECT_EQ(SP32(7u << 29, -33), getQuotient32(7, 16));

  // Divide evenly.
  EXPECT_EQ(SP32(3u << 30, -30), getQuotient32(9, 3));
  EXPECT_EQ(SP32(9u << 28, -28), getQuotient32(63, 7));

  // Divide unevenly.
  EXPECT_EQ(SP32(0xaaaaaaab, -33), getQuotient32(1, 3));
  EXPECT_EQ(SP32(0xd5555555, -31), getQuotient32(5, 3));

  // 64-bit division is hard to test, since divide64 doesn't canonicalized its
  // output.  However, this is the algorithm the implementation uses:
  //
  // - Shift divisor right.
  // - If we have 1 (power of 2), return early -- not canonicalized.
  // - Shift dividend left.
  // - 64-bit integer divide.
  // - If there's a remainder, continue with long division.
  //
  // TODO: require less knowledge about the implementation in the test.

  // Zero.
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 0));
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 1));
  EXPECT_EQ(SP64(0, 0), getQuotient64(0, 73));
  EXPECT_EQ(SP64(UINT64_MAX, INT16_MAX), getQuotient64(1, 0));
  EXPECT_EQ(SP64(UINT64_MAX, INT16_MAX), getQuotient64(6, 0));

  // Powers of two.
  EXPECT_EQ(SP64(1, 0), getQuotient64(1, 1));
  EXPECT_EQ(SP64(2, 0), getQuotient64(2, 1));
  EXPECT_EQ(SP64(4, -4), getQuotient64(4, 16));
  EXPECT_EQ(SP64(7, 0), getQuotient64(7, 1));
  EXPECT_EQ(SP64(7, -1), getQuotient64(7, 2));
  EXPECT_EQ(SP64(7, -4), getQuotient64(7, 16));

  // Divide evenly.
  EXPECT_EQ(SP64(UINT64_C(3) << 60, -60), getQuotient64(9, 3));
  EXPECT_EQ(SP64(UINT64_C(9) << 58, -58), getQuotient64(63, 7));

  // Divide unevenly.
  EXPECT_EQ(SP64(0xaaaaaaaaaaaaaaab, -65), getQuotient64(1, 3));
  EXPECT_EQ(SP64(0xd555555555555555, -63), getQuotient64(5, 3));
}

} // end namespace

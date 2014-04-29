//===- unittest/Support/BranchProbabilityTest.cpp - BranchProbability tests -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
void PrintTo(const BranchProbability &P, ::std::ostream *os) {
  *os << P.getNumerator() << "/" << P.getDenominator();
}
}
namespace {

typedef BranchProbability BP;
TEST(BranchProbabilityTest, Accessors) {
  EXPECT_EQ(1u, BP(1, 7).getNumerator());
  EXPECT_EQ(7u, BP(1, 7).getDenominator());
  EXPECT_EQ(0u, BP::getZero().getNumerator());
  EXPECT_EQ(1u, BP::getZero().getDenominator());
  EXPECT_EQ(1u, BP::getOne().getNumerator());
  EXPECT_EQ(1u, BP::getOne().getDenominator());
}

TEST(BranchProbabilityTest, Operators) {
  EXPECT_TRUE(BP(1, 7) < BP(2, 7));
  EXPECT_TRUE(BP(1, 7) < BP(1, 4));
  EXPECT_TRUE(BP(5, 7) < BP(3, 4));
  EXPECT_FALSE(BP(1, 7) < BP(1, 7));
  EXPECT_FALSE(BP(1, 7) < BP(2, 14));
  EXPECT_FALSE(BP(4, 7) < BP(1, 2));
  EXPECT_FALSE(BP(4, 7) < BP(3, 7));

  EXPECT_FALSE(BP(1, 7) > BP(2, 7));
  EXPECT_FALSE(BP(1, 7) > BP(1, 4));
  EXPECT_FALSE(BP(5, 7) > BP(3, 4));
  EXPECT_FALSE(BP(1, 7) > BP(1, 7));
  EXPECT_FALSE(BP(1, 7) > BP(2, 14));
  EXPECT_TRUE(BP(4, 7) > BP(1, 2));
  EXPECT_TRUE(BP(4, 7) > BP(3, 7));

  EXPECT_TRUE(BP(1, 7) <= BP(2, 7));
  EXPECT_TRUE(BP(1, 7) <= BP(1, 4));
  EXPECT_TRUE(BP(5, 7) <= BP(3, 4));
  EXPECT_TRUE(BP(1, 7) <= BP(1, 7));
  EXPECT_TRUE(BP(1, 7) <= BP(2, 14));
  EXPECT_FALSE(BP(4, 7) <= BP(1, 2));
  EXPECT_FALSE(BP(4, 7) <= BP(3, 7));

  EXPECT_FALSE(BP(1, 7) >= BP(2, 7));
  EXPECT_FALSE(BP(1, 7) >= BP(1, 4));
  EXPECT_FALSE(BP(5, 7) >= BP(3, 4));
  EXPECT_TRUE(BP(1, 7) >= BP(1, 7));
  EXPECT_TRUE(BP(1, 7) >= BP(2, 14));
  EXPECT_TRUE(BP(4, 7) >= BP(1, 2));
  EXPECT_TRUE(BP(4, 7) >= BP(3, 7));

  EXPECT_FALSE(BP(1, 7) == BP(2, 7));
  EXPECT_FALSE(BP(1, 7) == BP(1, 4));
  EXPECT_FALSE(BP(5, 7) == BP(3, 4));
  EXPECT_TRUE(BP(1, 7) == BP(1, 7));
  EXPECT_TRUE(BP(1, 7) == BP(2, 14));
  EXPECT_FALSE(BP(4, 7) == BP(1, 2));
  EXPECT_FALSE(BP(4, 7) == BP(3, 7));

  EXPECT_TRUE(BP(1, 7) != BP(2, 7));
  EXPECT_TRUE(BP(1, 7) != BP(1, 4));
  EXPECT_TRUE(BP(5, 7) != BP(3, 4));
  EXPECT_FALSE(BP(1, 7) != BP(1, 7));
  EXPECT_FALSE(BP(1, 7) != BP(2, 14));
  EXPECT_TRUE(BP(4, 7) != BP(1, 2));
  EXPECT_TRUE(BP(4, 7) != BP(3, 7));
}

TEST(BranchProbabilityTest, getCompl) {
  EXPECT_EQ(BP(5, 7), BP(2, 7).getCompl());
  EXPECT_EQ(BP(2, 7), BP(5, 7).getCompl());
  EXPECT_EQ(BP::getZero(), BP(7, 7).getCompl());
  EXPECT_EQ(BP::getOne(), BP(0, 7).getCompl());
}

TEST(BranchProbabilityTest, scale) {
  // Multiply by 1.0.
  EXPECT_EQ(UINT64_MAX, BP(1, 1).scale(UINT64_MAX));
  EXPECT_EQ(UINT64_MAX, BP(7, 7).scale(UINT64_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, 1).scale(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(7, 7).scale(UINT32_MAX));
  EXPECT_EQ(0u, BP(1, 1).scale(0));
  EXPECT_EQ(0u, BP(7, 7).scale(0));

  // Multiply by 0.0.
  EXPECT_EQ(0u, BP(0, 1).scale(UINT64_MAX));
  EXPECT_EQ(0u, BP(0, 1).scale(UINT64_MAX));
  EXPECT_EQ(0u, BP(0, 1).scale(0));

  auto Two63 = UINT64_C(1) << 63;
  auto Two31 = UINT64_C(1) << 31;

  // Multiply by 0.5.
  EXPECT_EQ(Two63 - 1, BP(1, 2).scale(UINT64_MAX));

  // Big fractions.
  EXPECT_EQ(1u, BP(Two31, UINT32_MAX).scale(2));
  EXPECT_EQ(Two31, BP(Two31, UINT32_MAX).scale(Two31 * 2));
  EXPECT_EQ(Two63 + Two31, BP(Two31, UINT32_MAX).scale(UINT64_MAX));

  // High precision.
  EXPECT_EQ(UINT64_C(9223372047592194055),
            BP(Two31 + 1, UINT32_MAX - 2).scale(UINT64_MAX));
}

TEST(BranchProbabilityTest, scaleByInverse) {
  // Divide by 1.0.
  EXPECT_EQ(UINT64_MAX, BP(1, 1).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(UINT64_MAX, BP(7, 7).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, 1).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(7, 7).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(0u, BP(1, 1).scaleByInverse(0));
  EXPECT_EQ(0u, BP(7, 7).scaleByInverse(0));

  // Divide by something very small.
  EXPECT_EQ(UINT64_MAX, BP(1, UINT32_MAX).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(uint64_t(UINT32_MAX) * UINT32_MAX,
            BP(1, UINT32_MAX).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, UINT32_MAX).scaleByInverse(1));

  auto Two63 = UINT64_C(1) << 63;
  auto Two31 = UINT64_C(1) << 31;

  // Divide by 0.5.
  EXPECT_EQ(UINT64_MAX - 1, BP(1, 2).scaleByInverse(Two63 - 1));
  EXPECT_EQ(UINT64_MAX, BP(1, 2).scaleByInverse(Two63));

  // Big fractions.
  EXPECT_EQ(1u, BP(Two31, UINT32_MAX).scaleByInverse(1));
  EXPECT_EQ(2u, BP(Two31 - 1, UINT32_MAX).scaleByInverse(1));
  EXPECT_EQ(Two31 * 2 - 1, BP(Two31, UINT32_MAX).scaleByInverse(Two31));
  EXPECT_EQ(Two31 * 2 + 1, BP(Two31 - 1, UINT32_MAX).scaleByInverse(Two31));
  EXPECT_EQ(UINT64_MAX, BP(Two31, UINT32_MAX).scaleByInverse(Two63 + Two31));

  // High precision.  The exact answers to these are close to the successors of
  // the floor.  If we were rounding, these would round up.
  EXPECT_EQ(UINT64_C(18446744065119617030),
            BP(Two31 + 2, UINT32_MAX - 2)
                .scaleByInverse(UINT64_C(9223372047592194055)));
  EXPECT_EQ(UINT64_C(18446744065119617026),
            BP(Two31 + 1, UINT32_MAX).scaleByInverse(Two63 + Two31));
}

}

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

}

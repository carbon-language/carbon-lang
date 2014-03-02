//===- llvm/unittest/ADT/APSIntTest.cpp - APSInt unit tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(APSIntTest, MoveTest) {
  APSInt A(32, true);
  EXPECT_TRUE(A.isUnsigned());

  APSInt B(128, false);
  A = B;
  EXPECT_FALSE(A.isUnsigned());

  APSInt C(B);
  EXPECT_FALSE(C.isUnsigned());

  APInt Wide(256, 0);
  const uint64_t *Bits = Wide.getRawData();
  APSInt D(std::move(Wide));
  EXPECT_TRUE(D.isUnsigned());
  EXPECT_EQ(Bits, D.getRawData()); // Verify that "Wide" was really moved.

  A = APSInt(64, true);
  EXPECT_TRUE(A.isUnsigned());

  Wide = APInt(128, 1);
  Bits = Wide.getRawData();
  A = std::move(Wide);
  EXPECT_TRUE(A.isUnsigned());
  EXPECT_EQ(Bits, A.getRawData()); // Verify that "Wide" was really moved.
}

}

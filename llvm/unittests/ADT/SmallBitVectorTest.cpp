//===- llvm/unittest/ADT/SmallBitVectorTest.cpp - SmallBitVector tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallBitVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SmallBitVectorTest, TrivialOperation) {
  SmallBitVector Vec;
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(0U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_TRUE(Vec.none());
  EXPECT_TRUE(Vec.empty());

  Vec.resize(5, true);
  EXPECT_EQ(5U, Vec.count());
  EXPECT_EQ(5U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Vec.resize(11);
  EXPECT_EQ(5U, Vec.count());
  EXPECT_EQ(11U, Vec.size());
  EXPECT_TRUE(Vec.any());
  EXPECT_FALSE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  SmallBitVector Inv = ~Vec;
  EXPECT_EQ(6U, Inv.count());
  EXPECT_EQ(11U, Inv.size());
  EXPECT_TRUE(Inv.any());
  EXPECT_FALSE(Inv.none());
  EXPECT_FALSE(Inv.empty());

  EXPECT_FALSE(Inv == Vec);
  EXPECT_TRUE(Inv != Vec);
  Vec = ~Vec;
  EXPECT_TRUE(Inv == Vec);
  EXPECT_FALSE(Inv != Vec);

  // Add some "interesting" data to Vec.
  Vec.resize(23, true);
  Vec.resize(25, false);
  Vec.resize(26, true);
  Vec.resize(29, false);
  Vec.resize(33, true);
  Vec.resize(61, false);
  unsigned Count = 0;
  for (unsigned i = Vec.find_first(); i != -1u; i = Vec.find_next(i)) {
    ++Count;
    EXPECT_TRUE(Vec[i]);
    EXPECT_TRUE(Vec.test(i));
  }
  EXPECT_EQ(Count, Vec.count());
  EXPECT_EQ(Count, 23u);
  EXPECT_FALSE(Vec[0]);
  EXPECT_TRUE(Vec[32]);
  EXPECT_FALSE(Vec[60]);

  SmallBitVector Copy = Vec;
  SmallBitVector Alt(3, false);
  Alt.resize(6, true);
  std::swap(Alt, Vec);
  EXPECT_TRUE(Copy == Alt);
  EXPECT_TRUE(Vec.size() == 6);
  EXPECT_TRUE(Vec.count() == 3);
  EXPECT_TRUE(Vec.find_first() == 3);
  std::swap(Copy, Vec);

  // Add some more "interesting" data.
  Vec.resize(68, true);
  Vec.resize(78, false);
  Vec.resize(89, true);
  Vec.resize(90, false);
  Vec.resize(91, true);
  Vec.resize(130, false);
  Count = 0;
  for (unsigned i = Vec.find_first(); i != -1u; i = Vec.find_next(i)) {
    ++Count;
    EXPECT_TRUE(Vec[i]);
    EXPECT_TRUE(Vec.test(i));
  }
  EXPECT_EQ(Count, Vec.count());
  EXPECT_EQ(Count, 42u);
  EXPECT_FALSE(Vec[0]);
  EXPECT_TRUE(Vec[32]);
  EXPECT_FALSE(Vec[60]);
  EXPECT_FALSE(Vec[129]);

  Vec.flip(60);
  EXPECT_TRUE(Vec[60]);
  EXPECT_EQ(Count + 1, Vec.count());
  Vec.flip(60);
  EXPECT_FALSE(Vec[60]);
  EXPECT_EQ(Count, Vec.count());

  Vec.reset(32);
  EXPECT_FALSE(Vec[32]);
  EXPECT_EQ(Count - 1, Vec.count());
  Vec.set(32);
  EXPECT_TRUE(Vec[32]);
  EXPECT_EQ(Count, Vec.count());

  Vec.flip();
  EXPECT_EQ(Vec.size() - Count, Vec.count());

  Vec.reset();
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(130U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_TRUE(Vec.none());
  EXPECT_FALSE(Vec.empty());

  Inv = ~SmallBitVector();
  EXPECT_EQ(0U, Inv.count());
  EXPECT_EQ(0U, Inv.size());
  EXPECT_FALSE(Inv.any());
  EXPECT_TRUE(Inv.none());
  EXPECT_TRUE(Inv.empty());

  Vec.clear();
  EXPECT_EQ(0U, Vec.count());
  EXPECT_EQ(0U, Vec.size());
  EXPECT_FALSE(Vec.any());
  EXPECT_TRUE(Vec.none());
  EXPECT_TRUE(Vec.empty());
}

}

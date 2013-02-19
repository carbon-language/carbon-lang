//===- unittest/ADT/MapVectorTest.cpp - MapVector unit tests ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/MapVector.h"
#include <utility>

using namespace llvm;

TEST(MapVectorTest, insert_pop) {
  MapVector<int, int> MV;
  std::pair<MapVector<int, int>::iterator, bool> R;

  R = MV.insert(std::make_pair(1, 2));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_TRUE(R.second);

  R = MV.insert(std::make_pair(1, 3));
  ASSERT_EQ(R.first, MV.begin());
  EXPECT_EQ(R.first->first, 1);
  EXPECT_EQ(R.first->second, 2);
  EXPECT_FALSE(R.second);

  R = MV.insert(std::make_pair(4, 5));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 5);
  EXPECT_TRUE(R.second);

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 5);

  MV.pop_back();
  EXPECT_EQ(MV.size(), 1u);
  EXPECT_EQ(MV[1], 2);

  R = MV.insert(std::make_pair(4, 7));
  ASSERT_NE(R.first, MV.end());
  EXPECT_EQ(R.first->first, 4);
  EXPECT_EQ(R.first->second, 7);
  EXPECT_TRUE(R.second);  

  EXPECT_EQ(MV.size(), 2u);
  EXPECT_EQ(MV[1], 2);
  EXPECT_EQ(MV[4], 7);
}

//===- llvm/unittest/ADT/DenseSetTest.cpp - DenseSet unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;

namespace {

// Test fixture
class DenseSetTest : public testing::Test {
};

// Test hashing with a set of only two entries.
TEST_F(DenseSetTest, DoubleEntrySetTest) {
  llvm::DenseSet<unsigned> set(2);
  set.insert(0);
  set.insert(1);
  // Original failure was an infinite loop in this call:
  EXPECT_EQ(0u, set.count(2));
}

struct TestDenseSetInfo {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37U; }
  static unsigned getHashValue(const char* Val) {
    return (unsigned)(Val[0] - 'a') * 37U;
  }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const char* LHS, const unsigned& RHS) {
    return (unsigned)(LHS[0] - 'a') == RHS;
  }
};

TEST(DenseSetCustomTest, FindAsTest) {
  DenseSet<unsigned, TestDenseSetInfo> set;
  set.insert(0);
  set.insert(1);
  set.insert(2);

  // Size tests
  EXPECT_EQ(3u, set.size());

  // Normal lookup tests
  EXPECT_EQ(1u, set.count(1));
  EXPECT_EQ(0u, *set.find(0));
  EXPECT_EQ(1u, *set.find(1));
  EXPECT_EQ(2u, *set.find(2));
  EXPECT_TRUE(set.find(3) == set.end());

  // find_as() tests
  EXPECT_EQ(0u, *set.find_as("a"));
  EXPECT_EQ(1u, *set.find_as("b"));
  EXPECT_EQ(2u, *set.find_as("c"));
  EXPECT_TRUE(set.find_as("d") == set.end());
}

}

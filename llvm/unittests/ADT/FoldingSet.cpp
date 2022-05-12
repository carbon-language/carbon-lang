//===- llvm/unittest/ADT/FoldingSetTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FoldingSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

// Unaligned string test.
TEST(FoldingSetTest, UnalignedStringTest) {
  SCOPED_TRACE("UnalignedStringTest");

  FoldingSetNodeID a, b;
  // An aligned string.
  std::string str1= "a test string";
  a.AddString(str1);

  // An unaligned string.
  std::string str2 = ">" + str1;
  b.AddString(str2.c_str() + 1);

  EXPECT_EQ(a.ComputeHash(), b.ComputeHash());
}

TEST(FoldingSetTest, LongLongComparison) {
  struct LongLongContainer : FoldingSetNode {
    unsigned long long A, B;
    LongLongContainer(unsigned long long A, unsigned long long B)
        : A(A), B(B) {}
    void Profile(FoldingSetNodeID &ID) const {
      ID.AddInteger(A);
      ID.AddInteger(B);
    }
  };

  LongLongContainer C1((1ULL << 32) + 1, 1ULL);
  LongLongContainer C2(1ULL, (1ULL << 32) + 1);

  FoldingSet<LongLongContainer> Set;

  EXPECT_EQ(&C1, Set.GetOrInsertNode(&C1));
  EXPECT_EQ(&C2, Set.GetOrInsertNode(&C2));
  EXPECT_EQ(2U, Set.size());
}

struct TrivialPair : public FoldingSetNode {
  unsigned Key = 0;
  unsigned Value = 0;
  TrivialPair(unsigned K, unsigned V) : FoldingSetNode(), Key(K), Value(V) {}

  void Profile(FoldingSetNodeID &ID) const {
    ID.AddInteger(Key);
    ID.AddInteger(Value);
  }
};

TEST(FoldingSetTest, IDComparison) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);

  void *InsertPos = nullptr;
  FoldingSetNodeID ID;
  T.Profile(ID);
  TrivialPair *N = Trivial.FindNodeOrInsertPos(ID, InsertPos);
  EXPECT_EQ(&T, N);
  EXPECT_EQ(nullptr, InsertPos);
}

TEST(FoldingSetTest, MissedIDComparison) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair S(100, 42);
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);

  void *InsertPos = nullptr;
  FoldingSetNodeID ID;
  S.Profile(ID);
  TrivialPair *N = Trivial.FindNodeOrInsertPos(ID, InsertPos);
  EXPECT_EQ(nullptr, N);
  EXPECT_NE(nullptr, InsertPos);
}

TEST(FoldingSetTest, RemoveNodeThatIsPresent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  EXPECT_EQ(Trivial.size(), 1U);

  bool WasThere = Trivial.RemoveNode(&T);
  EXPECT_TRUE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, RemoveNodeThatIsAbsent) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  bool WasThere = Trivial.RemoveNode(&T);
  EXPECT_FALSE(WasThere);
  EXPECT_EQ(0U, Trivial.size());
}

TEST(FoldingSetTest, GetOrInsertInserting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair *N = Trivial.GetOrInsertNode(&T);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, GetOrInsertGetting) {
  FoldingSet<TrivialPair> Trivial;

  TrivialPair T(99, 42);
  TrivialPair T2(99, 42);
  Trivial.InsertNode(&T);
  TrivialPair *N = Trivial.GetOrInsertNode(&T2);
  EXPECT_EQ(&T, N);
}

TEST(FoldingSetTest, InsertAtPos) {
  FoldingSet<TrivialPair> Trivial;

  void *InsertPos = nullptr;
  TrivialPair Finder(99, 42);
  FoldingSetNodeID ID;
  Finder.Profile(ID);
  Trivial.FindNodeOrInsertPos(ID, InsertPos);

  TrivialPair T(99, 42);
  Trivial.InsertNode(&T, InsertPos);
  EXPECT_EQ(1U, Trivial.size());
}

TEST(FoldingSetTest, EmptyIsTrue) {
  FoldingSet<TrivialPair> Trivial;
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, EmptyIsFalse) {
  FoldingSet<TrivialPair> Trivial;
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  EXPECT_FALSE(Trivial.empty());
}

TEST(FoldingSetTest, ClearOnEmpty) {
  FoldingSet<TrivialPair> Trivial;
  Trivial.clear();
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, ClearOnNonEmpty) {
  FoldingSet<TrivialPair> Trivial;
  TrivialPair T(99, 42);
  Trivial.InsertNode(&T);
  Trivial.clear();
  EXPECT_TRUE(Trivial.empty());
}

TEST(FoldingSetTest, CapacityLargerThanReserve) {
  FoldingSet<TrivialPair> Trivial;
  auto OldCapacity = Trivial.capacity();
  Trivial.reserve(OldCapacity + 1);
  EXPECT_GE(Trivial.capacity(), OldCapacity + 1);
}

TEST(FoldingSetTest, SmallReserveChangesNothing) {
  FoldingSet<TrivialPair> Trivial;
  auto OldCapacity = Trivial.capacity();
  Trivial.reserve(OldCapacity - 1);
  EXPECT_EQ(Trivial.capacity(), OldCapacity);
}

}


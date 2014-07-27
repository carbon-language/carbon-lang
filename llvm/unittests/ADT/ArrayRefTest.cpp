//===- llvm/unittest/ADT/ArrayRefTest.cpp - ArrayRef unit tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace llvm {

TEST(ArrayRefTest, AllocatorCopy) {
  BumpPtrAllocator Alloc;
  static const uint16_t Words1[] = { 1, 4, 200, 37 };
  ArrayRef<uint16_t> Array1 = makeArrayRef(Words1, 4);
  static const uint16_t Words2[] = { 11, 4003, 67, 64000, 13 };
  ArrayRef<uint16_t> Array2 = makeArrayRef(Words2, 5);
  ArrayRef<uint16_t> Array1c = Array1.copy(Alloc);
  ArrayRef<uint16_t> Array2c = Array2.copy(Alloc);;
  EXPECT_TRUE(Array1.equals(Array1c));
  EXPECT_NE(Array1.data(), Array1c.data());
  EXPECT_TRUE(Array2.equals(Array2c));
  EXPECT_NE(Array2.data(), Array2c.data());
}

TEST(ArrayRefTest, DropBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(TheNumbers, AR1.size() - 1);
  EXPECT_TRUE(AR1.drop_back().equals(AR2));
}

TEST(ArrayRefTest, Equals) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  ArrayRef<int> AR1(A1);
  EXPECT_TRUE(AR1.equals(1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_FALSE(AR1.equals(8, 1, 2, 4, 5, 6, 6, 7));
  EXPECT_FALSE(AR1.equals(2, 4, 5, 6, 6, 7, 8, 1));
  EXPECT_FALSE(AR1.equals(0, 1, 2, 4, 5, 6, 6, 7));
  EXPECT_FALSE(AR1.equals(1, 2, 42, 4, 5, 6, 7, 8));
  EXPECT_FALSE(AR1.equals(42, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_FALSE(AR1.equals(1, 2, 3, 4, 5, 6, 7, 42));
  EXPECT_FALSE(AR1.equals(1, 2, 3, 4, 5, 6, 7));
  EXPECT_FALSE(AR1.equals(1, 2, 3, 4, 5, 6, 7, 8, 9));

  ArrayRef<int> AR1a = AR1.drop_back();
  EXPECT_TRUE(AR1a.equals(1, 2, 3, 4, 5, 6, 7));
  EXPECT_FALSE(AR1a.equals(1, 2, 3, 4, 5, 6, 7, 8));

  ArrayRef<int> AR1b = AR1a.slice(2, 4);
  EXPECT_TRUE(AR1b.equals(3, 4, 5, 6));
  EXPECT_FALSE(AR1b.equals(2, 3, 4, 5, 6));
  EXPECT_FALSE(AR1b.equals(3, 4, 5, 6, 7));
}

} // end anonymous namespace

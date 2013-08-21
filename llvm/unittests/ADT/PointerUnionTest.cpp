//===- llvm/unittest/ADT/PointerUnionTest.cpp - Optional unit tests -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/PointerUnion.h"
using namespace llvm;

namespace {

typedef PointerUnion<int*, float*> PU;

// Test fixture
class PointerUnionTest : public testing::Test {
};

float f = 3.14f;
int i = 42;

const PU a(&f);
const PU b(&i);
const PU n;

TEST_F(PointerUnionTest, Comparison) {
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(b != n);
  EXPECT_FALSE(b == n);
}

TEST_F(PointerUnionTest, Null) {
  EXPECT_FALSE(a.isNull());
  EXPECT_FALSE(b.isNull());
  EXPECT_TRUE(n.isNull());
  EXPECT_FALSE(!a);
  EXPECT_FALSE(!b);
  EXPECT_TRUE(!n);
  // workaround an issue with EXPECT macros and explicit bool
  EXPECT_TRUE((bool)a);
  EXPECT_TRUE((bool)b);
  EXPECT_FALSE(n);
}

TEST_F(PointerUnionTest, Is) {
  EXPECT_FALSE(a.is<int*>());
  EXPECT_TRUE(a.is<float*>());
  EXPECT_TRUE(b.is<int*>());
  EXPECT_FALSE(b.is<float*>());
  EXPECT_TRUE(n.is<int*>());
  EXPECT_FALSE(n.is<float*>());
}

TEST_F(PointerUnionTest, Get) {
  EXPECT_EQ(a.get<float*>(), &f);
  EXPECT_EQ(b.get<int*>(), &i);
  EXPECT_EQ(n.get<int*>(), (int*)0);
}

} // end anonymous namespace

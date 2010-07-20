//===---------- llvm/unittest/Support/Casting.cpp - Casting tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#define DEBUG_CAST_OPERATORS
#include "llvm/Support/Casting.h"

#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

const foo *null_foo = NULL;

extern bar &B1;
extern const bar *B2;
// test various configurations of const
const bar &B3 = B1;
const bar *const B4 = B2;

TEST(CastingTest, isa) {
  EXPECT_TRUE(isa<foo>(B1));
  EXPECT_TRUE(isa<foo>(B2));
  EXPECT_TRUE(isa<foo>(B3));
  EXPECT_TRUE(isa<foo>(B4));
}

TEST(CastingTest, cast) {
  foo &F1 = cast<foo>(B1);
  EXPECT_NE(&F1, null_foo);
  const foo *F3 = cast<foo>(B2);
  EXPECT_NE(F3, null_foo);
  const foo *F4 = cast<foo>(B2);
  EXPECT_NE(F4, null_foo);
  const foo &F8 = cast<foo>(B3);
  EXPECT_NE(&F8, null_foo);
  const foo *F9 = cast<foo>(B4);
  EXPECT_NE(F9, null_foo);
  foo *F10 = cast<foo>(fub());
  EXPECT_EQ(F10, null_foo);
}

TEST(CastingTest, cast_or_null) {
  const foo *F11 = cast_or_null<foo>(B2);
  EXPECT_NE(F11, null_foo);
  const foo *F12 = cast_or_null<foo>(B2);
  EXPECT_NE(F12, null_foo);
  const foo *F13 = cast_or_null<foo>(B4);
  EXPECT_NE(F13, null_foo);
  const foo *F14 = cast_or_null<foo>(fub());  // Shouldn't print.
  EXPECT_EQ(F14, null_foo);
}

bar B;
bar &B1 = B;
const bar *B2 = &B;
}  // anonymous namespace

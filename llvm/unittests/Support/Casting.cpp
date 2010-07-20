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

extern bar &B1;
extern const bar *B2;

TEST(CastingTest, isa) {
  // test various configurations of const
  const bar &B3 = B1;
  const bar *const B4 = B2;
  EXPECT_TRUE(isa<foo>(B1));
  EXPECT_TRUE(isa<foo>(B2));
  EXPECT_TRUE(isa<foo>(B3));
  EXPECT_TRUE(isa<foo>(B4));
}

bar B;
bar &B1 = B;
const bar *B2 = &B;
}  // anonymous namespace

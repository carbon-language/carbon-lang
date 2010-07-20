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

TEST(CastingTest, Basics) {
  EXPECT_TRUE(isa<foo>(B1));
}

bar B;
bar &B1 = B;
const bar *B2 = &B;
}  // anonymous namespace

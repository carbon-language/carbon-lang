//===- llvm/unittest/Support/ThreadLocalTest.cpp - Therad Local tests   ---===//
//
//		       The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringPool.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(PooledStringPtrTest, OperatorEquals) {
  StringPool pool;
  const PooledStringPtr a = pool.intern("a");
  const PooledStringPtr b = pool.intern("b");
  EXPECT_FALSE(a == b);
}

TEST(PooledStringPtrTest, OperatorNotEquals) {
  StringPool pool;
  const PooledStringPtr a = pool.intern("a");
  const PooledStringPtr b = pool.intern("b");
  EXPECT_TRUE(a != b);
}

}

//===- llvm/unittest/ADT/HashingTest.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hashing.h unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/Hashing.h"

using namespace llvm;

namespace {

TEST(HashingTest, EmptyHashTest) {
  GeneralHash Hash;
  ASSERT_EQ(0u, Hash.finish());
}

TEST(HashingTest, IntegerHashTest) {
  ASSERT_TRUE(GeneralHash().add(1).finish() == GeneralHash().add(1).finish());
  ASSERT_TRUE(GeneralHash().add(1).finish() != GeneralHash().add(2).finish());
}

TEST(HashingTest, StringHashTest) {
  ASSERT_TRUE(
    GeneralHash().add("abc").finish() == GeneralHash().add("abc").finish());
  ASSERT_TRUE(
    GeneralHash().add("abc").finish() != GeneralHash().add("abcd").finish());
}

TEST(HashingTest, FloatHashTest) {
  ASSERT_TRUE(
    GeneralHash().add(1.0f).finish() == GeneralHash().add(1.0f).finish());
  ASSERT_TRUE(
    GeneralHash().add(1.0f).finish() != GeneralHash().add(2.0f).finish());
}

TEST(HashingTest, DoubleHashTest) {
  ASSERT_TRUE(GeneralHash().add(1.).finish() == GeneralHash().add(1.).finish());
  ASSERT_TRUE(GeneralHash().add(1.).finish() != GeneralHash().add(2.).finish());
}

TEST(HashingTest, IntegerArrayHashTest) {
  int a[] = { 1, 2 };
  int b[] = { 1, 3 };
  ASSERT_TRUE(GeneralHash().add(a).finish() == GeneralHash().add(a).finish());
  ASSERT_TRUE(GeneralHash().add(a).finish() != GeneralHash().add(b).finish());
}

}

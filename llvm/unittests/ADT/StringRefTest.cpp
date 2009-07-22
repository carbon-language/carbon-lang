//===- llvm/unittest/ADT/StringRefTest.cpp - StringRef unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

TEST(StringRefTest, Construction) {
  EXPECT_TRUE(StringRef() == "");
  EXPECT_TRUE(StringRef("hello") == "hello");
  EXPECT_TRUE(StringRef("hello world", 5) == "hello");
  EXPECT_TRUE(StringRef(std::string("hello")) == "hello");
}

TEST(StringRefTest, Iteration) {
  StringRef S("hello");
  const char *p = "hello";
  for (const char *it = S.begin(), *ie = S.end(); it != ie; ++it, ++p)
    EXPECT_TRUE(*p == *it);
}

TEST(StringRefTest, StringOps) {
  const char *p = "hello";
  EXPECT_EQ(p, StringRef(p, 0).data());
  EXPECT_TRUE(StringRef().empty());
  EXPECT_EQ((size_t) 5, StringRef("hello").size());
  EXPECT_EQ(-1, StringRef("aab").compare("aad"));
  EXPECT_EQ( 0, StringRef("aab").compare("aab"));
  EXPECT_EQ( 1, StringRef("aab").compare("aaa"));
  EXPECT_EQ(-1, StringRef("aab").compare("aabb"));
  EXPECT_EQ( 1, StringRef("aab").compare("aa"));
}

TEST(StringRefTest, Operators) {
  EXPECT_TRUE(StringRef() == "");
  EXPECT_TRUE(StringRef("aab") < StringRef("aad"));
  EXPECT_FALSE(StringRef("aab") < StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") <= StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") <= StringRef("aaa"));
  EXPECT_TRUE(StringRef("aad") > StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") > StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") >= StringRef("aab"));
  EXPECT_FALSE(StringRef("aaa") >= StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") == StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") == StringRef("aac"));
  EXPECT_FALSE(StringRef("aab") != StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") != StringRef("aac"));
  EXPECT_EQ('a', StringRef("aab")[1]);
}

TEST(StringRefTest, Utilities) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.substr(3) == "lo");
  EXPECT_TRUE(Str.substr(100) == "");
  EXPECT_TRUE(Str.substr(0, 100) == "hello");
  EXPECT_TRUE(Str.substr(4, 10) == "o");

  EXPECT_TRUE(Str.startswith("he"));
  EXPECT_FALSE(Str.startswith("helloworld"));
  EXPECT_FALSE(Str.startswith("hi"));

  std::string Storage;
  raw_string_ostream OS(Storage);
  OS << StringRef("hello");
  EXPECT_EQ("hello", OS.str());
}

} // end anonymous namespace

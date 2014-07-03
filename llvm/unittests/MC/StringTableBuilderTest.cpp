//===----------- StringTableBuilderTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

TEST(StringTableBuilderTest, Basic) {
  StringTableBuilder B;

  B.add("foo");
  B.add("bar");
  B.add("foobar");

  B.finalize();

  std::string Expected;
  Expected += '\x00';
  Expected += "foobar";
  Expected += '\x00';
  Expected += "foo";
  Expected += '\x00';

  EXPECT_EQ(Expected, B.data());
  EXPECT_EQ(1U, B.getOffset("foobar"));
  EXPECT_EQ(4U, B.getOffset("bar"));
  EXPECT_EQ(8U, B.getOffset("foo"));
}

}

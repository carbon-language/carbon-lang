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
#include "llvm/Support/Endian.h"
#include <string>

using namespace llvm;

namespace {

TEST(StringTableBuilderTest, BasicELF) {
  StringTableBuilder B;

  B.add("foo");
  B.add("bar");
  B.add("foobar");

  B.finalize(StringTableBuilder::ELF);

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

TEST(StringTableBuilderTest, BasicWinCOFF) {
  StringTableBuilder B;

  // Strings must be 9 chars or longer to go in the table.
  B.add("hippopotamus");
  B.add("pygmy hippopotamus");
  B.add("river horse");

  B.finalize(StringTableBuilder::WinCOFF);

  // size_field + "pygmy hippopotamus\0" + "river horse\0"
  uint32_t ExpectedSize = 4 + 19 + 12;
  EXPECT_EQ(ExpectedSize, B.data().size());

  std::string Expected;

  ExpectedSize =
      support::endian::byte_swap<uint32_t, support::little>(ExpectedSize);
  Expected.append((const char*)&ExpectedSize, 4);
  Expected += "pygmy hippopotamus";
  Expected += '\x00';
  Expected += "river horse";
  Expected += '\x00';

  EXPECT_EQ(Expected, B.data());
  EXPECT_EQ(4U, B.getOffset("pygmy hippopotamus"));
  EXPECT_EQ(10U, B.getOffset("hippopotamus"));
  EXPECT_EQ(23U, B.getOffset("river horse"));
}

}

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/output_buffer_ref.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/SmallString.h"

namespace Carbon::Terminal {
namespace {

using ::testing::Eq;

// A single piece and several pieces are appended by different code, so both
// appear throughout these tests rather than in one case of their own.

TEST(OutputBufferRefTest, Text) {
  llvm::SmallString<16> bytes;
  OutputBufferRef out = bytes;
  out.Append("one");
  out.Append(llvm::StringRef(" two"));
  out.Append(std::string(" three"));
  out.Append(" four", llvm::StringRef(" five"));
  EXPECT_THAT(bytes, Eq("one two three four five"));
}

TEST(OutputBufferRefTest, EmptyPieces) {
  llvm::SmallString<16> bytes;
  OutputBufferRef out = bytes;
  out.Append();
  out.Append("");
  out.Append("", llvm::StringRef(), "kept", llvm::StringRef(""));
  EXPECT_THAT(bytes, Eq("kept"));
}

TEST(OutputBufferRefTest, NumbersUseEveryDigitCount) {
  llvm::SmallString<16> bytes;
  OutputBufferRef out = bytes;
  for (uint8_t value : {0, 9, 10, 99, 100, 255}) {
    out.Append(value);
    out.Append(" ", value, " ");
  }
  EXPECT_THAT(bytes, Eq("0 0 9 9 10 10 99 99 100 100 255 255 "));
}

// A number takes fewer bytes than it reserves unless it has three digits, so
// pieces after one in the same call are what catch a misplaced write.
TEST(OutputBufferRefTest, NumbersFollowedByMorePieces) {
  llvm::SmallString<32> bytes;
  OutputBufferRef out = bytes;
  out.Append("\x1b[", static_cast<uint8_t>(38), ";2;", static_cast<uint8_t>(1),
             ";", static_cast<uint8_t>(22), ";", static_cast<uint8_t>(255),
             "m");
  EXPECT_THAT(bytes, Eq("\x1b[38;2;1;22;255m"));
}

TEST(OutputBufferRefTest, AppendsAfterExistingContents) {
  llvm::SmallString<16> bytes = llvm::StringRef("before:");
  OutputBufferRef out = bytes;
  out.Append(static_cast<uint8_t>(7));
  EXPECT_THAT(bytes, Eq("before:7"));
}

// Appending has to work the same however the buffer is laid out, and a number
// leaves the buffer grown further than it wrote, so reallocation is where a
// size mistake would show up.
TEST(OutputBufferRefTest, AppendsPastInlineCapacity) {
  llvm::SmallString<8> bytes;
  OutputBufferRef out = bytes;
  std::string expected;
  for (int i = 0; i < 100; ++i) {
    out.Append("x", static_cast<uint8_t>(i));
    expected += "x" + std::to_string(i);
  }
  EXPECT_THAT(bytes, Eq(expected));
  EXPECT_THAT(bytes.size(), Eq(expected.size()));
}

// References to one buffer all append to it, and none of them own it, so the
// buffer keeps everything written through any of them.
TEST(OutputBufferRefTest, ReferencesShareTheirBuffer) {
  llvm::SmallString<16> bytes;
  OutputBufferRef first = bytes;
  first.Append("a");
  {
    OutputBufferRef second = bytes;
    second.Append("b");
  }
  OutputBufferRef copy = first;
  copy.Append("c");
  first.Append("d");
  EXPECT_THAT(bytes, Eq("abcd"));
}

}  // namespace
}  // namespace Carbon::Terminal

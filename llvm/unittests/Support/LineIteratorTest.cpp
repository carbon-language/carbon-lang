//===- LineIterator.cpp - Unit tests --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

namespace {

TEST(LineIteratorTest, Basic) {
  std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer("line 1\n"
                                                                  "line 2\n"
                                                                  "line 3"));

  line_iterator I = line_iterator(*Buffer), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("line 3", *I);
  EXPECT_EQ(3, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, CommentSkipping) {
  std::unique_ptr<MemoryBuffer> Buffer(
      MemoryBuffer::getMemBuffer("line 1\n"
                                 "line 2\n"
                                 "# Comment 1\n"
                                 "line 4\n"
                                 "# Comment 2"));

  line_iterator I = line_iterator(*Buffer, '#'), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(1, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(2, I.line_number());
  ++I;
  EXPECT_EQ("line 4", *I);
  EXPECT_EQ(4, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, BlankSkipping) {
  std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer("\n\n\n"
                                                                  "line 1\n"
                                                                  "\n\n\n"
                                                                  "line 2\n"
                                                                  "\n\n\n"));

  line_iterator I = line_iterator(*Buffer), E;

  EXPECT_FALSE(I.is_at_eof());
  EXPECT_NE(E, I);

  EXPECT_EQ("line 1", *I);
  EXPECT_EQ(4, I.line_number());
  ++I;
  EXPECT_EQ("line 2", *I);
  EXPECT_EQ(8, I.line_number());
  ++I;

  EXPECT_TRUE(I.is_at_eof());
  EXPECT_EQ(E, I);
}

TEST(LineIteratorTest, EmptyBuffers) {
  std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer(""));
  EXPECT_TRUE(line_iterator(*Buffer).is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer));

  Buffer.reset(MemoryBuffer::getMemBuffer("\n\n\n"));
  EXPECT_TRUE(line_iterator(*Buffer).is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer));

  Buffer.reset(MemoryBuffer::getMemBuffer("# foo\n"
                                          "\n"
                                          "# bar"));
  EXPECT_TRUE(line_iterator(*Buffer, '#').is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer, '#'));

  Buffer.reset(MemoryBuffer::getMemBuffer("\n"
                                          "# baz\n"
                                          "\n"));
  EXPECT_TRUE(line_iterator(*Buffer, '#').is_at_eof());
  EXPECT_EQ(line_iterator(), line_iterator(*Buffer, '#'));
}

} // anonymous namespace

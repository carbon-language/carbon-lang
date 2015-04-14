//===- raw_pwrite_stream_test.cpp - raw_pwrite_stream tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

TEST(raw_pwrite_ostreamTest, TestSVector) {
  SmallString<64> Buffer;
  raw_svector_ostream OS(Buffer);
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(Test, OS.str());
}
}

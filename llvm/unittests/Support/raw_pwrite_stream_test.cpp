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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

TEST(raw_pwrite_ostreamTest, TestSVector) {
  SmallVector<char, 0> Buffer;
  raw_svector_ostream OS(Buffer);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(Test, OS.str());

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif
}

TEST(raw_pwrite_ostreamTest, TestFD) {
  SmallString<64> Path;
  int FD;
  sys::fs::createTemporaryFile("foo", "bar", FD, Path);
  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif
}

#ifdef LLVM_ON_UNIX
TEST(raw_pwrite_ostreamTest, TestDevNull) {
  int FD;
  sys::fs::openFileForWrite("/dev/null", FD, sys::fs::F_None);
  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);
}
#endif
}

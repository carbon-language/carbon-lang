//===- llvm/unittest/Support/MD5Test.cpp - MD5 tests ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MD5 functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MD5.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
void TestMD5Sum(ArrayRef<unsigned char> Input, StringRef Final) {
  MD5 Hash;
  Hash.update(Input);
  MD5::MD5Result MD5Res;
  Hash.final(MD5Res);
  SmallString<32> Res;
  MD5::stringifyResult(MD5Res, Res);
  EXPECT_EQ(Res, Final);
}

TEST(MD5Test, MD5) {
  TestMD5Sum(ArrayRef<unsigned char>((const unsigned char *)"", (size_t) 0),
             "d41d8cd98f00b204e9800998ecf8427e");
  TestMD5Sum(ArrayRef<unsigned char>((const unsigned char *)"a", (size_t) 1),
             "0cc175b9c0f1b6a831c399e269772661");
  TestMD5Sum(ArrayRef<unsigned char>(
                 (const unsigned char *)"abcdefghijklmnopqrstuvwxyz",
                 (size_t) 26),
             "c3fcd3d76192e4007dfb496cca67e13b");
}
}

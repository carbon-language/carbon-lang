//===- llvm/unittest/Support/ConvertUTFTest.cpp - ConvertUTF tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ConvertUTF.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

TEST(ConvertUTFTest, ConvertUTF16LittleEndianToUTF8String) {
  // Src is the look of disapproval.
  static const char Src[] = "\xff\xfe\xa0\x0c_\x00\xa0\x0c";
  ArrayRef<char> Ref(Src, sizeof(Src) - 1);
  std::string Result;
  bool Success = convertUTF16ToUTF8String(Ref, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, ConvertUTF16BigEndianToUTF8String) {
  // Src is the look of disapproval.
  static const char Src[] = "\xfe\xff\x0c\xa0\x00_\x0c\xa0";
  ArrayRef<char> Ref(Src, sizeof(Src) - 1);
  std::string Result;
  bool Success = convertUTF16ToUTF8String(Ref, Result);
  EXPECT_TRUE(Success);
  std::string Expected("\xe0\xb2\xa0_\xe0\xb2\xa0");
  EXPECT_EQ(Expected, Result);
}

TEST(ConvertUTFTest, OddLengthInput) {
  std::string Result;
  bool Success = convertUTF16ToUTF8String(ArrayRef<char>("xxxxx", 5), Result);
  EXPECT_FALSE(Success);
}

TEST(ConvertUTFTest, Empty) {
  std::string Result;
  bool Success = convertUTF16ToUTF8String(ArrayRef<char>(), Result);
  EXPECT_TRUE(Success);
  EXPECT_TRUE(Result.empty());
}

TEST(ConvertUTFTest, HasUTF16BOM) {
  bool HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>("\xff\xfe", 2));
  EXPECT_TRUE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>("\xfe\xff", 2));
  EXPECT_TRUE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>("\xfe\xff ", 3));
  EXPECT_TRUE(HasBOM); // Don't care about odd lengths.
  HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>("\xfe\xff\x00asdf", 6));
  EXPECT_TRUE(HasBOM);

  HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>());
  EXPECT_FALSE(HasBOM);
  HasBOM = hasUTF16ByteOrderMark(ArrayRef<char>("\xfe", 1));
  EXPECT_FALSE(HasBOM);
}

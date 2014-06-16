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
#include <vector>

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

std::pair<ConversionResult, std::vector<unsigned>>
ConvertUTF8ToUnicodeScalarsLenient(StringRef S) {
  const UTF8 *SourceStart = reinterpret_cast<const UTF8 *>(S.data());

  const UTF8 *SourceNext = SourceStart;
  std::vector<UTF32> Decoded(S.size(), 0);
  UTF32 *TargetStart = Decoded.data();

  auto Result =
      ConvertUTF8toUTF32(&SourceNext, SourceStart + S.size(), &TargetStart,
                         Decoded.data() + Decoded.size(), lenientConversion);

  Decoded.resize(TargetStart - Decoded.data());

  return std::make_pair(Result, Decoded);
}

#define R0(RESULT) std::make_pair(RESULT, std::vector<unsigned>{})
#define R(RESULT, ...) std::make_pair(RESULT, std::vector<unsigned>{ __VA_ARGS__ })

TEST(ConvertUTFTest, UTF8ToUTF32Lenient) {

  //
  // 1-byte sequences
  //

  // U+0041 LATIN CAPITAL LETTER A
  EXPECT_EQ(R(conversionOK, 0x0041),
      ConvertUTF8ToUnicodeScalarsLenient("\x41"));

  //
  // 2-byte sequences
  //

  // U+0283 LATIN SMALL LETTER ESH
  EXPECT_EQ(R(conversionOK, 0x0283),
      ConvertUTF8ToUnicodeScalarsLenient("\xca\x83"));

  // U+03BA GREEK SMALL LETTER KAPPA
  // U+1F79 GREEK SMALL LETTER OMICRON WITH OXIA
  // U+03C3 GREEK SMALL LETTER SIGMA
  // U+03BC GREEK SMALL LETTER MU
  // U+03B5 GREEK SMALL LETTER EPSILON
  EXPECT_EQ(R(conversionOK, 0x03ba, 0x1f79, 0x03c3, 0x03bc, 0x03b5),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce\xb5"));

  //
  // 3-byte sequences
  //

  // U+4F8B CJK UNIFIED IDEOGRAPH-4F8B
  // U+6587 CJK UNIFIED IDEOGRAPH-6587
  EXPECT_EQ(R(conversionOK, 0x4f8b, 0x6587),
      ConvertUTF8ToUnicodeScalarsLenient("\xe4\xbe\x8b\xe6\x96\x87"));

  // U+D55C HANGUL SYLLABLE HAN
  // U+AE00 HANGUL SYLLABLE GEUL
  EXPECT_EQ(R(conversionOK, 0xd55c, 0xae00),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\x95\x9c\xea\xb8\x80"));

  // U+1112 HANGUL CHOSEONG HIEUH
  // U+1161 HANGUL JUNGSEONG A
  // U+11AB HANGUL JONGSEONG NIEUN
  // U+1100 HANGUL CHOSEONG KIYEOK
  // U+1173 HANGUL JUNGSEONG EU
  // U+11AF HANGUL JONGSEONG RIEUL
  EXPECT_EQ(R(conversionOK, 0x1112, 0x1161, 0x11ab, 0x1100, 0x1173, 0x11af),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xe1\x84\x92\xe1\x85\xa1\xe1\x86\xab\xe1\x84\x80\xe1\x85\xb3"
          "\xe1\x86\xaf"));

  //
  // 4-byte sequences
  //

  // U+E0100 VARIATION SELECTOR-17
  EXPECT_EQ(R(conversionOK, 0x000E0100),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xa0\x84\x80"));

  //
  // First possible sequence of a certain length
  //

  // U+0000 NULL
  EXPECT_EQ(R(conversionOK, 0x0000),
      ConvertUTF8ToUnicodeScalarsLenient(StringRef("\x00", 1)));

  // U+0080 PADDING CHARACTER
  EXPECT_EQ(R(conversionOK, 0x0080),
      ConvertUTF8ToUnicodeScalarsLenient("\xc2\x80"));

  // U+0800 SAMARITAN LETTER ALAF
  EXPECT_EQ(R(conversionOK, 0x0800),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\xa0\x80"));

  // U+10000 LINEAR B SYLLABLE B008 A
  EXPECT_EQ(R(conversionOK, 0x10000),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x90\x80\x80"));

  // U+200000 (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x88\x80\x80\x80"));

  // U+4000000 (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x84\x80\x80\x80\x80"));

  //
  // Last possible sequence of a certain length
  //

  // U+007F DELETE
  EXPECT_EQ(R(conversionOK, 0x007f),
      ConvertUTF8ToUnicodeScalarsLenient("\x7f"));

  // U+07FF (unassigned)
  EXPECT_EQ(R(conversionOK, 0x07ff),
      ConvertUTF8ToUnicodeScalarsLenient("\xdf\xbf"));

  // U+FFFF (noncharacter)
  EXPECT_EQ(R(conversionOK, 0xffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xbf\xbf"));

  // U+1FFFFF (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7\xbf\xbf\xbf"));

  // U+3FFFFFF (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\xbf\xbf\xbf\xbf"));

  // U+7FFFFFFF (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\xbf\xbf\xbf\xbf\xbf"));

  //
  // Other boundary conditions
  //

  // U+D7FF (unassigned)
  EXPECT_EQ(R(conversionOK, 0xd7ff),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\x9f\xbf"));

  // U+E000 (private use)
  EXPECT_EQ(R(conversionOK, 0xe000),
      ConvertUTF8ToUnicodeScalarsLenient("\xee\x80\x80"));

  // U+FFFD REPLACEMENT CHARACTER
  EXPECT_EQ(R(conversionOK, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xbf\xbd"));

  // U+10FFFF (noncharacter)
  EXPECT_EQ(R(conversionOK, 0x10ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x8f\xbf\xbf"));

  // U+110000 (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x90\x80\x80"));

  //
  // Unexpected continuation bytes
  //

  // A sequence of unexpected continuation bytes that don't follow a first
  // byte, every byte is a maximal subpart.

  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xbf\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80\xbf\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80\xbf\x80\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\x80\xbf\x82\xbf\xaa"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xaa\xb0\xbb\xbf\xaa\xa0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xaa\xb0\xbb\xbf\xaa\xa0\x8f"));

  // All continuation bytes (0x80--0xbf).
  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
          "\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
          "\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
          "\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"));

  //
  // Lonely start bytes
  //

  // Start bytes of 2-byte sequences (0xc0--0xdf).
  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
          "\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"));

  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xc0\x20\xc1\x20\xc2\x20\xc3\x20\xc4\x20\xc5\x20\xc6\x20\xc7\x20"
          "\xc8\x20\xc9\x20\xca\x20\xcb\x20\xcc\x20\xcd\x20\xce\x20\xcf\x20"
          "\xd0\x20\xd1\x20\xd2\x20\xd3\x20\xd4\x20\xd5\x20\xd6\x20\xd7\x20"
          "\xd8\x20\xd9\x20\xda\x20\xdb\x20\xdc\x20\xdd\x20\xde\x20\xdf\x20"));

  // Start bytes of 3-byte sequences (0xe0--0xef).
  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"));

  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xe0\x20\xe1\x20\xe2\x20\xe3\x20\xe4\x20\xe5\x20\xe6\x20\xe7\x20"
          "\xe8\x20\xe9\x20\xea\x20\xeb\x20\xec\x20\xed\x20\xee\x20\xef\x20"));

  // Start bytes of 4-byte sequences (0xf0--0xf7).
  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7"));

  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xf0\x20\xf1\x20\xf2\x20\xf3\x20\xf4\x20\xf5\x20\xf6\x20\xf7\x20"));

  // Start bytes of 5-byte sequences (0xf8--0xfb).
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\xf9\xfa\xfb"));

  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x20\xf9\x20\xfa\x20\xfb\x20"));

  // Start bytes of 6-byte sequences (0xfc--0xfd).
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\xfd"));

  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x20\xfd\x20"));

  //
  // Other bytes (0xc0--0xc1, 0xfe--0xff).
  //

  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc1"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfe"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xff"));

  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0\xc1\xfe\xff"));

  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfe\xfe\xff\xff"));

  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfe\x80\x80\x80\x80\x80"));

  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xff\x80\x80\x80\x80\x80"));

  EXPECT_EQ(R(sourceIllegal,
              0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020, 0xfffd, 0x0020),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0\x20\xc1\x20\xfe\x20\xff\x20"));

  //
  // Sequences with one continuation byte missing
  //

  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc2"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xdf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\xa0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe1\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xec\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\x9f"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xee\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x90\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x8f\xbf"));

  // Overlong sequences with one trailing byte missing.
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc1"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\x9f"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x8f\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80\x80\x80\x80"));

  // Sequences that represent surrogates with one trailing byte missing.
  // High surrogates
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xac"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xaf"));
  // Low surrogates
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xb0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xb4"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xbf"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+1100xx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x90\x80"));
  // U+13FBxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf5\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf6\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7\x80\x80"));
  // U+1FFBxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7\xbf\xbf"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+2000xx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x88\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\xbf\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf9\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfa\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\x80\x80\x80"));
  // U+3FFFFxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\xbf\xbf\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10uzzzzz 10zzzyyyy 10yyyyxx 10xxxxxx
  // U+40000xx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x84\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\xbf\xbf\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\x80\x80\x80\x80"));
  // U+7FFFFFxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\xbf\xbf\xbf\xbf"));

  //
  // Sequences with two continuation bytes missing
  //

  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x90"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x8f"));

  // Overlong sequences with two trailing byte missing.
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x8f"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80\x80\x80"));

  // Sequences that represent surrogates with two trailing bytes missing.
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+110yxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x90"));
  // U+13Fyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf5\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf6\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7\x80"));
  // U+1FFyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7\xbf"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+200yxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x88\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf9\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfa\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\x80\x80"));
  // U+3FFFyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\xbf\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+4000yxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x84\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\xbf\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\x80\x80\x80"));
  // U+7FFFFyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\xbf\xbf\xbf"));

  //
  // Sequences with three continuation bytes missing
  //

  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4"));

  // Broken overlong sequences.
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80\x80"));

  // Ill-formed 4-byte sequences.
  // 11110zzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+14yyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf5"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf6"));
  // U+1Cyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf7"));

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+20yyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x88"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf9\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfa\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\x80"));
  // U+3FCyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb\xbf"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+400yyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x84\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\x80\x80"));
  // U+7FFCyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\xbf\xbf"));

  //
  // Sequences with four continuation bytes missing
  //

  // Ill-formed 5-byte sequences.
  // 111110uu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf9"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfa"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb"));
  // U+3zyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfb"));

  // Broken overlong sequences.
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80"));

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzzyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x84"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\x80"));
  // U+7Fzzyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd\xbf"));

  //
  // Sequences with five continuation bytes missing
  //

  // Ill-formed 6-byte sequences.
  // 1111110u 10uuuuuu 10zzzzzz 10zzyyyy 10yyyyxx 10xxxxxx
  // U+uzzyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc"));
  // U+uuzzyyxx (invalid)
  EXPECT_EQ(R(sourceIllegal, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfd"));

  //
  // Consecutive sequences with trailing bytes missing
  //

  EXPECT_EQ(R(sourceIllegal,
      0xfffd, /**/ 0xfffd, 0xfffd, /**/ 0xfffd, 0xfffd, 0xfffd, /**/
      0xfffd, 0xfffd, 0xfffd, 0xfffd,
      0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd,
      0xfffd, /**/ 0xfffd, /**/ 0xfffd, 0xfffd, 0xfffd, /**/
      0xfffd, 0xfffd, 0xfffd, 0xfffd,
      0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient(
          "\xc0" "\xe0\x80" "\xf0\x80\x80"
          "\xf8\x80\x80\x80"
          "\xfc\x80\x80\x80\x80"
          "\xdf" "\xef\xbf" "\xf7\xbf\xbf"
          "\xfb\xbf\xbf\xbf"
          "\xfd\xbf\xbf\xbf\xbf"));


  //
  // Overlong UTF-8 sequences
  //

  // U+002F SOLIDUS
  EXPECT_EQ(R(conversionOK, 0x002f),
      ConvertUTF8ToUnicodeScalarsLenient("\x2f"));

  // Overlong sequences of the above.
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0\xaf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\x80\xaf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x80\x80\xaf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x80\x80\x80\xaf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80\x80\x80\x80\xaf"));

  // U+0000 NULL
  EXPECT_EQ(R(conversionOK, 0x0000),
      ConvertUTF8ToUnicodeScalarsLenient(StringRef("\x00", 1)));

  // Overlong sequences of the above.
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x80\x80\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x80\x80\x80\x80\x80"));

  // Other overlong sequences.
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc0\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc1\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xc1\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xe0\x9f\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x8f\x80\x80"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x8f\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xf8\x87\xbf\xbf\xbf"));
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xfc\x83\xbf\xbf\xbf\xbf"));

  //
  // Isolated surrogates
  //

  // Unicode 6.3.0:
  //
  //    D71.  High-surrogate code point: A Unicode code point in the range
  //    U+D800 to U+DBFF.
  //
  //    D73.  Low-surrogate code point: A Unicode code point in the range
  //    U+DC00 to U+DFFF.

  // Note: U+E0100 is <DB40 DD00> in UTF16.

  // High surrogates

  // U+D800
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0\x80"));

  // U+DB40
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xac\xa0"));

  // U+DBFF
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xaf\xbf"));

  // Low surrogates

  // U+DC00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xb0\x80"));

  // U+DD00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xb4\x80"));

  // U+DFFF
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xbf\xbf"));

  // Surrogate pairs

  // U+D800 U+DC00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0\x80\xed\xb0\x80"));

  // U+D800 U+DD00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0\x80\xed\xb4\x80"));

  // U+D800 U+DFFF
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xa0\x80\xed\xbf\xbf"));

  // U+DB40 U+DC00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xac\xa0\xed\xb0\x80"));

  // U+DB40 U+DD00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xac\xa0\xed\xb4\x80"));

  // U+DB40 U+DFFF
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xac\xa0\xed\xbf\xbf"));

  // U+DBFF U+DC00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xaf\xbf\xed\xb0\x80"));

  // U+DBFF U+DD00
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xaf\xbf\xed\xb4\x80"));

  // U+DBFF U+DFFF
  EXPECT_EQ(R(sourceIllegal, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd, 0xfffd),
      ConvertUTF8ToUnicodeScalarsLenient("\xed\xaf\xbf\xed\xbf\xbf"));

  //
  // Noncharacters
  //

  // Unicode 6.3.0:
  //
  //    D14.  Noncharacter: A code point that is permanently reserved for
  //    internal use and that should never be interchanged. Noncharacters
  //    consist of the values U+nFFFE and U+nFFFF (where n is from 0 to 1016)
  //    and the values U+FDD0..U+FDEF.

  // U+FFFE
  EXPECT_EQ(R(conversionOK, 0xfffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xbf\xbe"));

  // U+FFFF
  EXPECT_EQ(R(conversionOK, 0xffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xbf\xbf"));

  // U+1FFFE
  EXPECT_EQ(R(conversionOK, 0x1fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x9f\xbf\xbe"));

  // U+1FFFF
  EXPECT_EQ(R(conversionOK, 0x1ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\x9f\xbf\xbf"));

  // U+2FFFE
  EXPECT_EQ(R(conversionOK, 0x2fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xaf\xbf\xbe"));

  // U+2FFFF
  EXPECT_EQ(R(conversionOK, 0x2ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xaf\xbf\xbf"));

  // U+3FFFE
  EXPECT_EQ(R(conversionOK, 0x3fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xbf\xbf\xbe"));

  // U+3FFFF
  EXPECT_EQ(R(conversionOK, 0x3ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf0\xbf\xbf\xbf"));

  // U+4FFFE
  EXPECT_EQ(R(conversionOK, 0x4fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x8f\xbf\xbe"));

  // U+4FFFF
  EXPECT_EQ(R(conversionOK, 0x4ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x8f\xbf\xbf"));

  // U+5FFFE
  EXPECT_EQ(R(conversionOK, 0x5fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x9f\xbf\xbe"));

  // U+5FFFF
  EXPECT_EQ(R(conversionOK, 0x5ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\x9f\xbf\xbf"));

  // U+6FFFE
  EXPECT_EQ(R(conversionOK, 0x6fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\xaf\xbf\xbe"));

  // U+6FFFF
  EXPECT_EQ(R(conversionOK, 0x6ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\xaf\xbf\xbf"));

  // U+7FFFE
  EXPECT_EQ(R(conversionOK, 0x7fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\xbf\xbf\xbe"));

  // U+7FFFF
  EXPECT_EQ(R(conversionOK, 0x7ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf1\xbf\xbf\xbf"));

  // U+8FFFE
  EXPECT_EQ(R(conversionOK, 0x8fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\x8f\xbf\xbe"));

  // U+8FFFF
  EXPECT_EQ(R(conversionOK, 0x8ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\x8f\xbf\xbf"));

  // U+9FFFE
  EXPECT_EQ(R(conversionOK, 0x9fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\x9f\xbf\xbe"));

  // U+9FFFF
  EXPECT_EQ(R(conversionOK, 0x9ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\x9f\xbf\xbf"));

  // U+AFFFE
  EXPECT_EQ(R(conversionOK, 0xafffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\xaf\xbf\xbe"));

  // U+AFFFF
  EXPECT_EQ(R(conversionOK, 0xaffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\xaf\xbf\xbf"));

  // U+BFFFE
  EXPECT_EQ(R(conversionOK, 0xbfffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\xbf\xbf\xbe"));

  // U+BFFFF
  EXPECT_EQ(R(conversionOK, 0xbffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf2\xbf\xbf\xbf"));

  // U+CFFFE
  EXPECT_EQ(R(conversionOK, 0xcfffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\x8f\xbf\xbe"));

  // U+CFFFF
  EXPECT_EQ(R(conversionOK, 0xcfffF),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\x8f\xbf\xbf"));

  // U+DFFFE
  EXPECT_EQ(R(conversionOK, 0xdfffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\x9f\xbf\xbe"));

  // U+DFFFF
  EXPECT_EQ(R(conversionOK, 0xdffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\x9f\xbf\xbf"));

  // U+EFFFE
  EXPECT_EQ(R(conversionOK, 0xefffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xaf\xbf\xbe"));

  // U+EFFFF
  EXPECT_EQ(R(conversionOK, 0xeffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xaf\xbf\xbf"));

  // U+FFFFE
  EXPECT_EQ(R(conversionOK, 0xffffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xbf\xbf\xbe"));

  // U+FFFFF
  EXPECT_EQ(R(conversionOK, 0xfffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf3\xbf\xbf\xbf"));

  // U+10FFFE
  EXPECT_EQ(R(conversionOK, 0x10fffe),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x8f\xbf\xbe"));

  // U+10FFFF
  EXPECT_EQ(R(conversionOK, 0x10ffff),
      ConvertUTF8ToUnicodeScalarsLenient("\xf4\x8f\xbf\xbf"));

  // U+FDD0
  EXPECT_EQ(R(conversionOK, 0xfdd0),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x90"));

  // U+FDD1
  EXPECT_EQ(R(conversionOK, 0xfdd1),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x91"));

  // U+FDD2
  EXPECT_EQ(R(conversionOK, 0xfdd2),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x92"));

  // U+FDD3
  EXPECT_EQ(R(conversionOK, 0xfdd3),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x93"));

  // U+FDD4
  EXPECT_EQ(R(conversionOK, 0xfdd4),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x94"));

  // U+FDD5
  EXPECT_EQ(R(conversionOK, 0xfdd5),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x95"));

  // U+FDD6
  EXPECT_EQ(R(conversionOK, 0xfdd6),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x96"));

  // U+FDD7
  EXPECT_EQ(R(conversionOK, 0xfdd7),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x97"));

  // U+FDD8
  EXPECT_EQ(R(conversionOK, 0xfdd8),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x98"));

  // U+FDD9
  EXPECT_EQ(R(conversionOK, 0xfdd9),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x99"));

  // U+FDDA
  EXPECT_EQ(R(conversionOK, 0xfdda),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9a"));

  // U+FDDB
  EXPECT_EQ(R(conversionOK, 0xfddb),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9b"));

  // U+FDDC
  EXPECT_EQ(R(conversionOK, 0xfddc),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9c"));

  // U+FDDD
  EXPECT_EQ(R(conversionOK, 0xfddd),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9d"));

  // U+FDDE
  EXPECT_EQ(R(conversionOK, 0xfdde),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9e"));

  // U+FDDF
  EXPECT_EQ(R(conversionOK, 0xfddf),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\x9f"));

  // U+FDE0
  EXPECT_EQ(R(conversionOK, 0xfde0),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa0"));

  // U+FDE1
  EXPECT_EQ(R(conversionOK, 0xfde1),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa1"));

  // U+FDE2
  EXPECT_EQ(R(conversionOK, 0xfde2),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa2"));

  // U+FDE3
  EXPECT_EQ(R(conversionOK, 0xfde3),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa3"));

  // U+FDE4
  EXPECT_EQ(R(conversionOK, 0xfde4),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa4"));

  // U+FDE5
  EXPECT_EQ(R(conversionOK, 0xfde5),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa5"));

  // U+FDE6
  EXPECT_EQ(R(conversionOK, 0xfde6),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa6"));

  // U+FDE7
  EXPECT_EQ(R(conversionOK, 0xfde7),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa7"));

  // U+FDE8
  EXPECT_EQ(R(conversionOK, 0xfde8),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa8"));

  // U+FDE9
  EXPECT_EQ(R(conversionOK, 0xfde9),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xa9"));

  // U+FDEA
  EXPECT_EQ(R(conversionOK, 0xfdea),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xaa"));

  // U+FDEB
  EXPECT_EQ(R(conversionOK, 0xfdeb),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xab"));

  // U+FDEC
  EXPECT_EQ(R(conversionOK, 0xfdec),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xac"));

  // U+FDED
  EXPECT_EQ(R(conversionOK, 0xfded),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xad"));

  // U+FDEE
  EXPECT_EQ(R(conversionOK, 0xfdee),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xae"));

  // U+FDEF
  EXPECT_EQ(R(conversionOK, 0xfdef),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xaf"));

  // U+FDF0
  EXPECT_EQ(R(conversionOK, 0xfdf0),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb0"));

  // U+FDF1
  EXPECT_EQ(R(conversionOK, 0xfdf1),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb1"));

  // U+FDF2
  EXPECT_EQ(R(conversionOK, 0xfdf2),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb2"));

  // U+FDF3
  EXPECT_EQ(R(conversionOK, 0xfdf3),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb3"));

  // U+FDF4
  EXPECT_EQ(R(conversionOK, 0xfdf4),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb4"));

  // U+FDF5
  EXPECT_EQ(R(conversionOK, 0xfdf5),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb5"));

  // U+FDF6
  EXPECT_EQ(R(conversionOK, 0xfdf6),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb6"));

  // U+FDF7
  EXPECT_EQ(R(conversionOK, 0xfdf7),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb7"));

  // U+FDF8
  EXPECT_EQ(R(conversionOK, 0xfdf8),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb8"));

  // U+FDF9
  EXPECT_EQ(R(conversionOK, 0xfdf9),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xb9"));

  // U+FDFA
  EXPECT_EQ(R(conversionOK, 0xfdfa),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xba"));

  // U+FDFB
  EXPECT_EQ(R(conversionOK, 0xfdfb),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xbb"));

  // U+FDFC
  EXPECT_EQ(R(conversionOK, 0xfdfc),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xbc"));

  // U+FDFD
  EXPECT_EQ(R(conversionOK, 0xfdfd),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xbd"));

  // U+FDFE
  EXPECT_EQ(R(conversionOK, 0xfdfe),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xbe"));

  // U+FDFF
  EXPECT_EQ(R(conversionOK, 0xfdff),
      ConvertUTF8ToUnicodeScalarsLenient("\xef\xb7\xbf"));
}

std::pair<ConversionResult, std::vector<unsigned>>
ConvertUTF8ToUnicodeScalarsPartialLenient(StringRef S) {
  const UTF8 *SourceStart = reinterpret_cast<const UTF8 *>(S.data());

  const UTF8 *SourceNext = SourceStart;
  std::vector<UTF32> Decoded(S.size(), 0);
  UTF32 *TargetStart = Decoded.data();

  auto Result = ConvertUTF8toUTF32Partial(
      &SourceNext, SourceStart + S.size(), &TargetStart,
      Decoded.data() + Decoded.size(), lenientConversion);

  Decoded.resize(TargetStart - Decoded.data());

  return std::make_pair(Result, Decoded);
}

TEST(ConvertUTFTest, UTF8ToUTF32PartialLenient) {
  // U+0041 LATIN CAPITAL LETTER A
  EXPECT_EQ(R(conversionOK, 0x0041),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\x41"));

  //
  // Sequences with one continuation byte missing
  //

  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xc2"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xdf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xe0\xa0"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xe0\xbf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xe1\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xec\xbf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xed\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xed\x9f"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xee\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xef\xbf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf0\x90\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf0\xbf\xbf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf1\x80\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf3\xbf\xbf"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf4\x80\x80"));
  EXPECT_EQ(R0(sourceExhausted),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\xf4\x8f\xbf"));

  EXPECT_EQ(R(sourceExhausted, 0x0041),
      ConvertUTF8ToUnicodeScalarsPartialLenient("\x41\xc2"));
}

#undef R0
#undef R


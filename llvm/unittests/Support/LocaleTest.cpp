//===- unittests/Support/LocaleTest.cpp - Locale.h tests ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Locale.h"
#include "gtest/gtest.h"

namespace llvm {
namespace sys {
namespace locale {
namespace {

// FIXME: WIN32 implementation is incorrect. We should consider using the one
// from LocaleGeneric.inc for WIN32.
#ifndef _WIN32
TEST(Locale, columnWidth) {
  // FIXME: This test fails with MacOSX implementation of columnWidth.
#ifndef __APPLE__
  EXPECT_EQ(0, columnWidth(""));
  EXPECT_EQ(1, columnWidth(" "));
  EXPECT_EQ(1, columnWidth("a"));
  EXPECT_EQ(1, columnWidth("~"));

  EXPECT_EQ(6, columnWidth("abcdef"));

  EXPECT_EQ(-1, columnWidth("\x01"));
  EXPECT_EQ(-1, columnWidth("aaaaaaaaaa\x01"));
  EXPECT_EQ(-1, columnWidth("\342\200\213")); // 200B ZERO WIDTH SPACE

  EXPECT_EQ(0, columnWidth("\314\200")); // 0300 COMBINING GRAVE ACCENT
  EXPECT_EQ(1, columnWidth("\340\270\201")); // 0E01 THAI CHARACTER KO KAI
  EXPECT_EQ(2, columnWidth("\344\270\200")); // CJK UNIFIED IDEOGRAPH-4E00

  EXPECT_EQ(4, columnWidth("\344\270\200\344\270\200"));
  EXPECT_EQ(3, columnWidth("q\344\270\200"));
  EXPECT_EQ(3, columnWidth("\314\200\340\270\201\344\270\200"));

  // Invalid UTF-8 strings, columnWidth should error out.
  EXPECT_EQ(-2, columnWidth("\344"));
  EXPECT_EQ(-2, columnWidth("\344\270"));
  EXPECT_EQ(-2, columnWidth("\344\270\033"));
  EXPECT_EQ(-2, columnWidth("\344\270\300"));
  EXPECT_EQ(-2, columnWidth("\377\366\355"));

  EXPECT_EQ(-2, columnWidth("qwer\344"));
  EXPECT_EQ(-2, columnWidth("qwer\344\270"));
  EXPECT_EQ(-2, columnWidth("qwer\344\270\033"));
  EXPECT_EQ(-2, columnWidth("qwer\344\270\300"));
  EXPECT_EQ(-2, columnWidth("qwer\377\366\355"));

  // UTF-8 sequences longer than 4 bytes correspond to unallocated Unicode
  // characters.
  EXPECT_EQ(-2, columnWidth("\370\200\200\200\200"));     // U+200000
  EXPECT_EQ(-2, columnWidth("\374\200\200\200\200\200")); // U+4000000
#endif // __APPLE__
}

TEST(Locale, isPrint) {
  EXPECT_EQ(false, isPrint(0)); // <control-0000>-<control-001F>
  EXPECT_EQ(false, isPrint(0x01));
  EXPECT_EQ(false, isPrint(0x1F));
  EXPECT_EQ(true, isPrint(' '));
  EXPECT_EQ(true, isPrint('A'));
  EXPECT_EQ(true, isPrint('~'));
  EXPECT_EQ(false, isPrint(0x7F)); // <control-007F>..<control-009F>
  EXPECT_EQ(false, isPrint(0x90));
  EXPECT_EQ(false, isPrint(0x9F));

  EXPECT_EQ(true, isPrint(0xAC));
  // FIXME: Figure out if we want to treat SOFT HYPHEN as printable character.
#ifndef __APPLE__
  EXPECT_EQ(false, isPrint(0xAD)); // SOFT HYPHEN
#endif // __APPLE__
  EXPECT_EQ(true, isPrint(0xAE));

  // MacOS implementation doesn't think it's printable.
#ifndef __APPLE__
  EXPECT_EQ(true, isPrint(0x0377)); // GREEK SMALL LETTER PAMPHYLIAN DIGAMMA
#endif // __APPLE__
  EXPECT_EQ(false, isPrint(0x0378)); // <reserved-0378>..<reserved-0379>

  EXPECT_EQ(false, isPrint(0x0600)); // ARABIC NUMBER SIGN

  EXPECT_EQ(false, isPrint(0x1FFFF)); // <reserved-1F774>..<noncharacter-1FFFF>
  EXPECT_EQ(true, isPrint(0x20000)); // CJK UNIFIED IDEOGRAPH-20000

  EXPECT_EQ(false, isPrint(0x10FFFF)); // noncharacter
}

#endif // _WIN32

} // namespace
} // namespace locale
} // namespace sys
} // namespace llvm

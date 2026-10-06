// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/metrics.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {
namespace {

// "e" followed by U+0301 COMBINING ACUTE ACCENT, which is one column because
// the mark renders into the column the "e" is in.
static constexpr llvm::StringLiteral AcuteE = "é";

TEST(MetricsTest, Width) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.Width(""), 0);
  EXPECT_EQ(utf8.Width("hello"), 5);
  EXPECT_EQ(utf8.Width("中中"), 4);
  EXPECT_EQ(utf8.Width("a中b"), 4);
  EXPECT_EQ(utf8.Width(AcuteE), 1);

  // Every byte is a column when the terminal isn't decoding UTF-8.
  Metrics ascii(Charset::Ascii);
  EXPECT_EQ(ascii.Width("hello"), 5);
  EXPECT_EQ(ascii.Width("中中"), 6);
  EXPECT_EQ(ascii.Width(AcuteE), 3);
}

TEST(MetricsTest, CodePointWidth) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.CodePointWidth(U'a'), 1);
  EXPECT_EQ(utf8.CodePointWidth(U'中'), 2);
  // A combining mark renders into the column before it.
  EXPECT_EQ(utf8.CodePointWidth(U'́'), 0);
  // Something with no rendering is drawn as a replacement, which is a column.
  EXPECT_EQ(utf8.CodePointWidth(U''), 1);

  Metrics ascii(Charset::Ascii);
  EXPECT_EQ(ascii.CodePointWidth(U'a'), 1);
  EXPECT_EQ(ascii.CodePointWidth(U'中'), 1);
  EXPECT_EQ(ascii.CodePointWidth(U'́'), 1);
}

TEST(MetricsTest, OnlyCombiningMarksAreZeroColumns) {
  // Drawing reads a width of zero as "renders into the cell before this one",
  // so a code point that takes no column without combining with anything has to
  // measure as something else. Terminals disagree about these -- Terminal.app
  // gives U+200C a column and VS Code's terminal gives it none -- so each is
  // drawn as a replacement character, which takes exactly one.
  Metrics utf8(Charset::Utf8);
  for (char32_t code_point : {U'\u200b', U'\u200c', U'\u200d', U'\ufeff'}) {
    EXPECT_EQ(utf8.CodePointWidth(code_point), 1)
        << static_cast<uint32_t>(code_point);
    EXPECT_EQ(utf8.RenderedCodePoint(code_point), U'�')
        << static_cast<uint32_t>(code_point);
  }
}

TEST(MetricsTest, RenderedCodePoint) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.RenderedCodePoint(U'a'), U'a');
  EXPECT_EQ(utf8.RenderedCodePoint(U'中'), U'中');
  EXPECT_EQ(utf8.RenderedCodePoint(U''), U'�');

  // Code points that UTF-8 has no encoding for have no rendering either.
  EXPECT_EQ(utf8.RenderedCodePoint(static_cast<char32_t>(0xd800)), U'�');
  EXPECT_EQ(utf8.RenderedCodePoint(static_cast<char32_t>(0x110000)), U'�');

  // An ASCII terminal is only given what it draws as itself, because there is
  // no telling what it would draw for anything else.
  Metrics ascii(Charset::Ascii);
  EXPECT_EQ(ascii.RenderedCodePoint(U'a'), U'a');
  EXPECT_EQ(ascii.RenderedCodePoint(U'中'), U'?');
  EXPECT_EQ(ascii.RenderedCodePoint(U''), U'?');
}

TEST(MetricsTest, TakeColumns) {
  Metrics utf8(Charset::Utf8);
  llvm::StringRef text = "abcde";
  EXPECT_EQ(utf8.TakeColumns(text, 3), "abc");
  EXPECT_EQ(text, "de");

  // Taking more than there is takes all of it.
  EXPECT_EQ(utf8.TakeColumns(text, 10), "de");
  EXPECT_EQ(text, "");

  // Taking nothing takes nothing, and a negative width is no different.
  text = "abcde";
  EXPECT_EQ(utf8.TakeColumns(text, 0), "");
  EXPECT_EQ(utf8.TakeColumns(text, -1), "");
  EXPECT_EQ(text, "abcde");
}

TEST(MetricsTest, TakeColumnsKeepsWideCharactersWhole) {
  Metrics utf8(Charset::Utf8);
  // A character that would straddle the end stops the walk before it, so the
  // prefix comes back a column short rather than half a character wide.
  llvm::StringRef text = "中中中";
  llvm::StringRef prefix = utf8.TakeColumns(text, 3);
  EXPECT_EQ(prefix, "中");
  EXPECT_EQ(utf8.Width(prefix), 2);
  EXPECT_EQ(text, "中中");

  // A request landing on a character boundary takes the whole prefix.
  text = "中中中";
  EXPECT_EQ(utf8.TakeColumns(text, 4), "中中");
  EXPECT_EQ(text, "中");
}

TEST(MetricsTest, TakeColumnsUnderAscii) {
  // Every byte is a column, so a multi-byte character is cut like any other
  // run of bytes.
  Metrics ascii(Charset::Ascii);
  llvm::StringRef text = "中";
  EXPECT_EQ(ascii.TakeColumns(text, 2).size(), 2U);
  EXPECT_EQ(text.size(), 1U);
}

TEST(MetricsTest, TakeCodePointResynchronizesOnInvalidUtf8) {
  Metrics utf8(Charset::Utf8);
  // A byte that starts no valid sequence is consumed on its own, so the text
  // after it is still decoded rather than being discarded.
  llvm::StringRef text =
      "\xff"
      "a";
  EXPECT_EQ(utf8.TakeCodePoint(text), U'�');
  EXPECT_EQ(utf8.TakeCodePoint(text), U'a');
  EXPECT_TRUE(text.empty());
}

TEST(MetricsTest, EncodeUtf8) {
  Utf8Storage storage;
  EXPECT_EQ(EncodeUtf8(U'a', storage), "a");
  EXPECT_EQ(EncodeUtf8(U'é', storage), "é");
  EXPECT_EQ(EncodeUtf8(U'中', storage), "中");
  EXPECT_EQ(EncodeUtf8(U'\U0001f525', storage), "\U0001f525");

  // A code point with no encoding of its own becomes the replacement.
  EXPECT_EQ(EncodeUtf8(static_cast<char32_t>(0xd800), storage), "�");
  EXPECT_EQ(EncodeUtf8(static_cast<char32_t>(0x110000), storage), "�");
}

TEST(MetricsDeathTest, WidthRejectsPositionalCharacters) {
  // A tab's width is a fact about a drawing rather than about the text, so
  // answering for one here would be answering a question this can't see the
  // inputs to.
  Metrics metrics(Charset::Utf8);
  EXPECT_DEATH((void)metrics.Width("a\tb"), "Width is only for text whose");
  EXPECT_DEATH((void)metrics.Width("a\nb"), "Width is only for text whose");
  EXPECT_DEATH((void)metrics.Width("a\rb"), "Width is only for text whose");
}

}  // namespace
}  // namespace Carbon::Terminal

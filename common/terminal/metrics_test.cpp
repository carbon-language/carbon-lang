// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/metrics.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

TEST(MetricsTest, SymbolWidth) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.SymbolWidth(U'a'), 1);
  EXPECT_EQ(utf8.SymbolWidth(U'中'), 2);
  // A combining mark renders into the column before it.
  EXPECT_EQ(utf8.SymbolWidth(U'́'), 0);
  // Something with no rendering is drawn as a replacement, which is a column.
  EXPECT_EQ(utf8.SymbolWidth(U''), 1);

  Metrics ascii(Charset::Ascii);
  EXPECT_EQ(ascii.SymbolWidth(U'a'), 1);
  EXPECT_EQ(ascii.SymbolWidth(U'中'), 1);
  EXPECT_EQ(ascii.SymbolWidth(U'́'), 1);
}

TEST(MetricsTest, RenderedSymbol) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.RenderedSymbol(U'a'), U'a');
  EXPECT_EQ(utf8.RenderedSymbol(U'中'), U'中');
  EXPECT_EQ(utf8.RenderedSymbol(U''), U'�');

  // An ASCII terminal is only given what it draws as itself, because there is
  // no telling what it would draw for anything else.
  Metrics ascii(Charset::Ascii);
  EXPECT_EQ(ascii.RenderedSymbol(U'a'), U'a');
  EXPECT_EQ(ascii.RenderedSymbol(U'中'), U'?');
  EXPECT_EQ(ascii.RenderedSymbol(U''), U'?');
}

TEST(MetricsTest, WrapWidth) {
  Metrics utf8(Charset::Utf8);
  EXPECT_EQ(utf8.WrapWidth(""), 0);
  EXPECT_EQ(utf8.WrapWidth("a bb ccc"), 3);
  EXPECT_EQ(utf8.WrapWidth("  spaced  out  "), 6);

  // Newlines and tabs bound a word as break characters, not as columns of
  // their own.
  EXPECT_EQ(utf8.WrapWidth("a\nbb\tccc"), 3);

  // A word is measured in the columns it takes, not the bytes it holds.
  EXPECT_EQ(utf8.WrapWidth("中中 a"), 4);
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

TEST(MetricsTest, TakeColumnsKeepsWideSymbolsWhole) {
  Metrics utf8(Charset::Utf8);
  // A symbol that would straddle the end stops the walk before it, so the
  // prefix comes back a column short rather than half a character wide.
  llvm::StringRef text = "中中中";
  llvm::StringRef prefix = utf8.TakeColumns(text, 3);
  EXPECT_EQ(prefix, "中");
  EXPECT_EQ(utf8.Width(prefix), 2);
  EXPECT_EQ(text, "中中");

  // Asking for exactly what one takes takes it.
  text = "中中中";
  EXPECT_EQ(utf8.TakeColumns(text, 4), "中中");
  EXPECT_EQ(text, "中");
}

TEST(MetricsTest, TakeColumnsUnderAscii) {
  // Every byte is a column, so a multi-byte character is cut like any other
  // run of bytes.
  Metrics ascii(Charset::Ascii);
  llvm::StringRef text = "中";
  EXPECT_EQ(ascii.TakeColumns(text, 2).size(), 2u);
  EXPECT_EQ(text.size(), 1u);
}

TEST(MetricsTest, TakeSymbolResynchronizesOnInvalidUtf8) {
  Metrics utf8(Charset::Utf8);
  // A byte that starts no valid sequence is consumed on its own, so the text
  // after it is still decoded rather than being discarded.
  llvm::StringRef text =
      "\xff"
      "a";
  EXPECT_EQ(utf8.TakeSymbol(text), U'�');
  EXPECT_EQ(utf8.TakeSymbol(text), U'a');
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

}  // namespace
}  // namespace Carbon::Terminal

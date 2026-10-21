// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adversarial inputs across the terminal library's surface.
//
// The library draws whatever a source file contains, for a terminal whose width
// came from an environment variable. Both are things a user controls, and
// neither may crash, read out of bounds, or quietly produce a position that is
// wrong. The tests here feed each entry point the inputs most likely to do one
// of those, and assert that what comes back is coherent rather than asserting
// any particular rendering.
//
// What is deliberately not here: anything a caller is checked for getting
// wrong, which is text past `MaxTextBytes` and coordinates outside the width
// laid out for. Those are death tests in `buffer_test`. What remains is the
// text and the width, neither of which a caller can validate ahead of drawing.

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "common/terminal/buffer.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/metrics.h"
#include "common/terminal/style.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {
namespace {

// Byte sequences that stress column accounting.
auto HostileText() -> std::vector<std::string> {
  return {
      "",
      "\x01\x02\x7f",               // C0 controls and delete
      "\xff\xfe\xfd",               // never valid UTF-8
      "\xe4\xb8",                   // truncated multi-byte sequence
      "\xe4\xb8\x96\xe4\xb8",       // valid then truncated
      "\xcc\x81",                   // combining mark with no base
      "e\xcc\x81\xcc\x82\xcc\x83",  // a base with several marks
      "\xf0\x9f\x94\xa5",           // outside the basic plane
      "\xed\xa0\x80",               // a surrogate, which UTF-8 forbids
      "\xc0\x80",                   // overlong encoding of NUL
      "中中中",                     // double-width throughout
      "a\tb\nc\r\nd",               // every positional character
      "\t\t\t\t\t\t\t\t",           // nothing but tabs
      "\n\n\n",                     // nothing but newlines
      std::string(4096, ' '),       // a long run of blanks
      std::string(1024, '\t'),      // a long run of tabs
  };
}

// Renders `buffer` and checks the bytes are coherent. Everything here draws
// with the default style, so what a style leaves behind is `buffer_test`'s to
// cover; this is about the text surviving at all.
auto RenderAndCheck(const Buffer& buffer, ColorMode mode) -> std::string {
  llvm::SmallString<256> out;
  buffer.Render(out, mode);
  std::string rendered(out);
  if (rendered.empty()) {
    return rendered;
  }
  EXPECT_EQ(rendered.back(), '\n');
  return rendered;
}

TEST(PressureTest, DrawTextSurvivesHostileBytes) {
  for (Charset charset : {Charset::Ascii, Charset::Utf8}) {
    for (const std::string& text : HostileText()) {
      for (int x : {0, 1, 7}) {
        Buffer buffer(8, charset);
        Buffer::DrawEnd end = buffer.DrawText(x, 0, text, Style());

        // The end is where drawing would carry on, which is not always a cell
        // that exists: text ending in a newline leaves it on a row nothing was
        // drawn on. Nor does the width follow from it, since the grid grows by
        // halves and overshoots. What must hold is that the end names a
        // non-negative cell, and that no row was created past where drawing
        // ended.
        EXPECT_GE(end.y, 0) << text;
        EXPECT_LE(buffer.height(), end.y + 1) << text;
        EXPECT_GE(end.x, 0) << text;

        // Measuring answers what drawing did.
        EXPECT_EQ(buffer.MeasureText(x, 0, text), end) << text;

        RenderAndCheck(buffer, ColorMode::Ansi16);
      }
    }
  }
}

TEST(PressureTest, WrappedTextSurvivesHostileBytes) {
  for (Charset charset : {Charset::Ascii, Charset::Utf8}) {
    for (const std::string& text : HostileText()) {
      // A width of one is the tightest anything can be asked to wrap into.
      for (int width : {1, 2, 3, 80}) {
        Buffer buffer(width, charset);
        Buffer::DrawEnd end =
            buffer.DrawWrappedText(0, 0, 0, width, text, Style());

        EXPECT_GE(end.y, 0) << text;
        EXPECT_LE(buffer.height(), end.y + 1) << text;
        EXPECT_GE(end.x, 0) << text;
        EXPECT_EQ(buffer.MeasureWrappedText(0, 0, 0, width, text), end) << text;
        // The width wrapping this wouldn't overhang is a fact about the text
        // rather than about the block it was drawn into.
        EXPECT_GE(buffer.MeasureWrapWidth(text), 0) << text;

        RenderAndCheck(buffer, ColorMode::Truecolor);
      }
    }
  }
}

TEST(PressureTest, MetricsSurviveHostileBytes) {
  for (Charset charset : {Charset::Ascii, Charset::Utf8}) {
    Metrics metrics(charset);
    for (const std::string& text : HostileText()) {
      // `Width` requires text with no positional characters in it, which is
      // checked, so only the rest is measured here.
      if (llvm::StringRef(text).find_first_of("\t\n\r") ==
          llvm::StringRef::npos) {
        int width = metrics.Width(text);
        EXPECT_GE(width, 0) << text;

        // Cutting at any column gives back a prefix that is no wider than
        // asked for and that leaves the rest of the string behind it.
        for (int columns : {-1, 0, 1, 2, width, width + 1}) {
          llvm::StringRef rest = text;
          llvm::StringRef prefix = metrics.TakeColumns(rest, columns);
          EXPECT_EQ(prefix.size() + rest.size(), text.size()) << text;
          EXPECT_LE(metrics.Width(prefix), std::max(columns, 0)) << text;
        }
      }

      // Taking code points consumes the whole string however invalid it is,
      // rather than stalling on a byte it can't decode.
      llvm::StringRef rest = text;
      size_t steps = 0;
      while (!rest.empty()) {
        metrics.TakeCodePoint(rest);
        ++steps;
        ASSERT_LE(steps, text.size()) << "TakeCodePoint failed to consume";
      }
    }
  }
}

TEST(PressureTest, OverlappingDrawsLeaveNoHalfCharacters) {
  // Double-width characters, lines, and text all writing over each other is
  // where a stale continuation cell would show up as a rendering with half a
  // character in it.
  Buffer buffer(8, Charset::Utf8);
  for (int pass = 0; pass < 3; ++pass) {
    buffer.DrawText(0, 0, "中中中中", Style());
    buffer.DrawHorizontalLine(1, 0, 3, Style());
    buffer.DrawText(2, 0, "中", Style());
    buffer.DrawVerticalLine(3, 0, 2, Style());
    buffer.DrawCodePoint(4, 0, U'中', Style());
    buffer.DrawCodePoint(5, 0, U'x', Style());
    buffer.DrawText(0, 0, "ab", Style());
  }

  // `Render` encodes every cell it emits, so this checks the overdraws leave a
  // grid that renders at all rather than that no half character survived.
  std::string rendered = RenderAndCheck(buffer, ColorMode::NoColor);
  Metrics metrics(Charset::Utf8);
  llvm::StringRef rest = rendered;
  while (!rest.empty()) {
    EXPECT_NE(metrics.TakeCodePoint(rest), 0xfffd) << rendered;
  }
}

TEST(PressureTest, CapabilitiesWidthNeverBreaksTheBuffer) {
  // A width claimed by the environment can be anything at all.
  for (int columns :
       {-1, 0, 1, 2, 80, Buffer::MaxColumns - 1, Buffer::MaxColumns,
        Buffer::MaxColumns + 1, 1 << 20, std::numeric_limits<int>::max()}) {
    Capabilities capabilities = {.charset = Charset::Utf8, .columns = columns};
    Buffer buffer(capabilities);
    // Whatever was claimed, what comes out is a width that can be laid out for
    // and drawn into.
    EXPECT_GE(buffer.columns(), 1) << columns;
    EXPECT_LE(buffer.columns(), Buffer::MaxColumns) << columns;
    buffer.DrawText(0, 0, "中x", Style());
    RenderAndCheck(buffer, ColorMode::Ansi256);
  }
}

}  // namespace
}  // namespace Carbon::Terminal

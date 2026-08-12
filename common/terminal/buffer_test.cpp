// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>

#include "common/filesystem.h"
#include "common/terminal/metrics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {
namespace {

// "e" followed by U+0301 COMBINING ACUTE ACCENT. Spelled out because the
// precomposed U+00E9 is a single code point and wouldn't exercise marks at all.
constexpr char32_t CombiningAcute = 0x0301;
constexpr llvm::StringLiteral AcuteE = "e\xcc\x81";

using DrawEnd = Buffer::DrawEnd;

auto Render(const Buffer& buffer, ColorMode mode = ColorMode::NoColor)
    -> std::string {
  llvm::SmallString<256> out;
  buffer.Render(out, mode);
  return std::string(out);
}

TEST(BufferTest, Empty) {
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.width(), 10);
  EXPECT_EQ(buffer.height(), 0);
  EXPECT_EQ(Render(buffer), "");
}

TEST(BufferTest, GrowsRowsToFitWhatIsDrawn) {
  Buffer buffer(4, Charset::Ascii);
  EXPECT_EQ(buffer.DrawCodePoint(0, 2, 'x', Style()).x, 1);
  EXPECT_EQ(buffer.height(), 3);
  EXPECT_EQ(Render(buffer), "\n\nx\n");
}

TEST(BufferTest, WidthIsTheTargetUntilSomethingOverhangs) {
  // `width()` tracks `columns()` while everything stays inside it.
  Buffer buffer(6, Charset::Ascii);
  EXPECT_EQ(buffer.columns(), 6);
  EXPECT_EQ(buffer.width(), 6);
  buffer.DrawHorizontalLine(1, 0, 5, Style());
  buffer.DrawVerticalLine(0, 1, 5, Style());
  EXPECT_EQ(buffer.width(), 6);
  EXPECT_EQ(Render(buffer),
            " -----\n"
            "|\n"
            "|\n"
            "|\n"
            "|\n"
            "|\n");

  // Text that starts inside the width can run past it, and a word wrapping
  // cannot break is the case that produces.
  Buffer overhang(4, Charset::Ascii);
  overhang.DrawWrappedText(0, 0, 0, 4, "ab wordthatislong", Style());
  EXPECT_EQ(overhang.columns(), 4);
  EXPECT_GE(overhang.width(), 14);
  EXPECT_EQ(Render(overhang),
            "ab\n"
            "wordthatislong\n");
}

TEST(BufferTest, WidthGrowsAcrossRowsAlreadyDrawn) {
  // Rows are stored back to back, so widening has to reflow the ones already
  // there rather than leaving them where their old width put them.
  Buffer buffer(2, Charset::Ascii);
  buffer.DrawText(0, 0, "ab", Style());
  buffer.DrawText(0, 1, "cd", Style());
  buffer.DrawText(0, 2, "efghij", Style());
  EXPECT_EQ(Render(buffer),
            "ab\n"
            "cd\n"
            "efghij\n");
}

TEST(BufferTest, GrowthFromEveryStartingWidth) {
  // Growth runs in steps, so every starting width reaches the same place by a
  // different route.
  for (int start : {1, 2, 3, 5, 17}) {
    Buffer buffer(start, Charset::Utf8);
    for (int row = 0; row < 4; ++row) {
      buffer.DrawText(0, row, "abcdefghijklmnopqrstuvwxyz", Style());
    }
    EXPECT_EQ(Render(buffer),
              "abcdefghijklmnopqrstuvwxyz\n"
              "abcdefghijklmnopqrstuvwxyz\n"
              "abcdefghijklmnopqrstuvwxyz\n"
              "abcdefghijklmnopqrstuvwxyz\n")
        << start;
  }
}

TEST(BufferTest, LinesJoinWhereTheyMeet) {
  Buffer buffer(3, Charset::Utf8);
  buffer.DrawHorizontalLine(0, 1, 3, Style());
  buffer.DrawVerticalLine(1, 0, 3, Style());

  EXPECT_EQ(Render(buffer),
            " ╷\n"
            "╶┼╴\n"
            " ╵\n");
}

TEST(BufferTest, LineEndsFormCorners) {
  Buffer buffer(3, Charset::Utf8);
  buffer.DrawHorizontalLine(1, 0, 2, Style());
  buffer.DrawVerticalLine(1, 0, 3, Style());

  EXPECT_EQ(Render(buffer),
            " ╭╴\n"
            " │\n"
            " ╵\n");
}

TEST(BufferTest, LinesOnlyJoinWhereTheyOverlap) {
  // A cell's glyph follows from the directions lines leave it in, so joining
  // is a matter of drawing into the same cell rather than of being adjacent.
  Buffer separate(3, Charset::Utf8);
  separate.DrawHorizontalLine(0, 0, 3, Style());
  separate.DrawVerticalLine(0, 1, 2, Style());
  EXPECT_EQ(Render(separate),
            "╶─╴\n"
            "╷\n"
            "╵\n");

  Buffer overlapping(3, Charset::Utf8);
  overlapping.DrawHorizontalLine(0, 0, 3, Style());
  overlapping.DrawVerticalLine(0, 0, 3, Style());
  EXPECT_EQ(Render(overlapping),
            "╭─╴\n"
            "│\n"
            "╵\n");
}

TEST(BufferTest, DrawOrderDoesNotMatter) {
  Buffer vertical_first(3, Charset::Utf8);
  vertical_first.DrawVerticalLine(1, 0, 3, Style());
  vertical_first.DrawHorizontalLine(0, 1, 3, Style());

  EXPECT_EQ(Render(vertical_first),
            " ╷\n"
            "╶┼╴\n"
            " ╵\n");
}

TEST(BufferTest, Tees) {
  // Every junction shape comes out of lines overlapping, so a table of them
  // exercises all four tees and the cross together.
  Buffer buffer(5, Charset::Utf8);
  buffer.DrawBox(0, 0, 5, 5, Style());
  buffer.DrawHorizontalLine(0, 2, 5, Style());
  buffer.DrawVerticalLine(2, 0, 5, Style());
  EXPECT_EQ(Render(buffer),
            "╭─┬─╮\n"
            "│ │ │\n"
            "├─┼─┤\n"
            "│ │ │\n"
            "╰─┴─╯\n");

  // The same table in ASCII, where every tee keeps its through-stroke and only
  // the crossing in the middle is a `+`.
  Buffer ascii(5, Charset::Ascii);
  ascii.DrawBox(0, 0, 5, 5, Style());
  ascii.DrawHorizontalLine(0, 2, 5, Style());
  ascii.DrawVerticalLine(2, 0, 5, Style());
  EXPECT_EQ(Render(ascii),
            ".---.\n"
            "| | |\n"
            "|-+-|\n"
            "| | |\n"
            "'---'\n");
}

TEST(BufferTest, ALineBetweenOneCenterAndItselfIsAPoint) {
  // A line of one cell between two centers has no length and no direction, so
  // it is a point. It is still line art, so anything drawn through it later
  // joins it rather than replacing it.
  Buffer horizontal(3, Charset::Utf8);
  horizontal.DrawHorizontalLine(1, 0, 1, Style());
  EXPECT_EQ(Render(horizontal), " ·\n");

  Buffer vertical(3, Charset::Utf8);
  vertical.DrawVerticalLine(1, 0, 1, Style());
  EXPECT_EQ(Render(vertical), " ·\n");

  Buffer joined(3, Charset::Utf8);
  joined.DrawHorizontalLine(1, 0, 1, Style());
  joined.DrawHorizontalLine(0, 0, 3, Style());
  EXPECT_EQ(Render(joined), "╶─╴\n");

  // A point is a small mark in either character set, rather than the junction
  // ASCII draws where lines really cross.
  Buffer ascii(3, Charset::Ascii);
  ascii.DrawVerticalLine(1, 0, 1, Style());
  EXPECT_EQ(Render(ascii), " .\n");

  // A line with no length draws nothing at all.
  Buffer empty(3, Charset::Utf8);
  empty.DrawHorizontalLine(0, 0, 0, Style());
  empty.DrawVerticalLine(0, 0, 0, Style());
  EXPECT_EQ(Render(empty), "");
}

TEST(BufferTest, LineEndsDecideWhatMeetsThemAtTheEnd) {
  // A line ending at a center is met with a corner, because the two lines stop
  // at the same point. One running out through the edge of its last cell is met
  // with a tee, because it carries on past whatever arrives there.
  Buffer corner(4, Charset::Utf8);
  corner.DrawHorizontalLine(0, 0, 3, Style());
  corner.DrawVerticalLine(2, 0, 2, Style());
  EXPECT_EQ(Render(corner),
            "╶─╮\n"
            "  ╵\n");

  Buffer tee(4, Charset::Utf8);
  tee.DrawHorizontalLine(0, 0, 3, Style(), LineEnd::Center, LineEnd::Edge);
  tee.DrawVerticalLine(2, 0, 2, Style());
  EXPECT_EQ(Render(tee),
            "╶─┬\n"
            "  ╵\n");

  // `start` decides the first cell the way `end` decides the last.
  Buffer edge_start(4, Charset::Utf8);
  edge_start.DrawHorizontalLine(1, 0, 2, Style(), LineEnd::Edge);
  EXPECT_EQ(Render(edge_start), " ─╴\n");
}

TEST(BufferTest, AnEndAtACenterDrawsHalfALine) {
  // Two half-lines laid end to end show the gap that says they do not connect,
  // which ASCII cannot draw.
  Buffer buffer(6, Charset::Utf8);
  buffer.DrawHorizontalLine(0, 0, 4, Style());
  buffer.DrawVerticalLine(0, 1, 3, Style());
  EXPECT_EQ(Render(buffer),
            "╶──╴\n"
            "╷\n"
            "│\n"
            "╵\n");

  // Two lines that each stop at their own center, laid end to end, are drawn
  // with the gap between them that says they do not connect.
  Buffer apart(6, Charset::Utf8);
  apart.DrawHorizontalLine(0, 0, 2, Style());
  apart.DrawHorizontalLine(2, 0, 2, Style());
  EXPECT_EQ(Render(apart), "╶╴╶╴\n");

  // ASCII has nothing to draw half a line with, so there the two read as one.
  Buffer ascii(6, Charset::Ascii);
  ascii.DrawHorizontalLine(0, 0, 2, Style());
  ascii.DrawHorizontalLine(2, 0, 2, Style());
  EXPECT_EQ(Render(ascii), "----\n");
}

TEST(BufferTest, AnEdgeToEdgeLineSpansWholeCells) {
  // Bounding a run of columns is edge to edge: the line covers all of them
  // rather than stopping halfway into the first and last.
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawHorizontalLine(0, 0, 1, Style(), LineEnd::Edge, LineEnd::Edge);
  buffer.DrawVerticalLine(0, 1, 1, Style(), LineEnd::Edge, LineEnd::Edge);
  EXPECT_EQ(Render(buffer),
            "─\n"
            "│\n");

  // Two edge-to-edge runs that meet end to end read as one line, without
  // either having to reach into the other's cells.
  Buffer split(6, Charset::Utf8);
  split.DrawVerticalLine(0, 0, 2, Style(), LineEnd::Edge, LineEnd::Edge);
  split.DrawVerticalLine(0, 2, 2, Style(), LineEnd::Edge, LineEnd::Edge);
  EXPECT_EQ(Render(split),
            "│\n"
            "│\n"
            "│\n"
            "│\n");
}

TEST(BufferTest, LinesDrawOverContent) {
  // A line replaces whatever text was in the cell, including both halves of a
  // double-width character it lands on.
  Buffer over_text(5, Charset::Utf8);
  over_text.DrawText(0, 0, "abcde", Style());
  over_text.DrawHorizontalLine(1, 0, 3, Style());
  EXPECT_EQ(Render(over_text), "a╶─╴e\n");

  Buffer over_wide(5, Charset::Utf8);
  over_wide.DrawText(0, 0, "中中", Style());
  over_wide.DrawVerticalLine(1, 0, 1, Style(), LineEnd::Edge, LineEnd::Edge);
  EXPECT_EQ(Render(over_wide), " │中\n");
}

TEST(BufferTest, Box) {
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawBox(0, 0, 4, 4, Style());
  EXPECT_EQ(Render(buffer),
            "╭──╮\n"
            "│  │\n"
            "│  │\n"
            "╰──╯\n");

  // The ASCII stand-ins keep the shape: the sides run and the corners turn,
  // with the character that sits low where the line leaves downward and the one
  // that sits high where it arrives from above.
  Buffer ascii(4, Charset::Ascii);
  ascii.DrawBox(0, 0, 4, 3, Style());
  EXPECT_EQ(Render(ascii),
            ".--.\n"
            "|  |\n"
            "'--'\n");

  // A box with no interior is the single line that bounds it.
  Buffer flat(4, Charset::Utf8);
  flat.DrawBox(0, 0, 4, 1, Style());
  flat.DrawBox(0, 2, 1, 2, Style());
  EXPECT_EQ(Render(flat),
            "╶──╴\n"
            "\n"
            "╷\n"
            "╵\n");

  Buffer degenerate(4, Charset::Utf8);
  degenerate.DrawBox(0, 0, 0, 4, Style());
  degenerate.DrawBox(0, 0, 4, 0, Style());
  EXPECT_EQ(Render(degenerate), "");
}

TEST(BufferTest, TextIsNeverRedrawnAsLines) {
  Buffer buffer(10, Charset::Utf8);
  buffer.DrawText(0, 0, "a -+- b", Style());
  EXPECT_EQ(Render(buffer), "a -+- b\n");
}

TEST(BufferTest, Text) {
  Buffer buffer(15, Charset::Utf8);
  EXPECT_EQ(buffer.DrawText(0, 0, "Hello, World!", Style()).y, 0);
  EXPECT_EQ(buffer.DrawText(0, 1, "A\tB\nC", Style()).y, 2);

  EXPECT_EQ(Render(buffer),
            "Hello, World!\n"
            "A       B\n"
            "C\n");

  // Drawing nothing ends where it began.
  EXPECT_EQ(buffer.DrawText(0, 0, "", Style()).y, 0);
}

TEST(BufferTest, TabStopsAreMeasuredFromWhereTextBegins) {
  // Tab stops follow the text's own origin, not the left edge, so a source
  // line quoted beside a gutter keeps the tab alignment it had in the file.
  Buffer buffer(20, Charset::Utf8);
  buffer.DrawText(3, 0, "A\tB", Style());
  EXPECT_EQ(Render(buffer), "   A       B\n");
}

TEST(BufferTest, TextRowsReturnToTheMargin) {
  // Text drawn as differently styled spans names one margin across all of them,
  // so a newline in the middle of it returns to the text's own left edge rather
  // than to wherever the span it fell in started.
  Buffer buffer(20, Charset::Ascii);
  buffer.DrawText(0, 0, "| ", Style());
  DrawEnd end = buffer.DrawText(2, 0, 2, "plain\nmore ", Style());
  buffer.DrawText(end.x, end.y, 2, "bold\ntext", Style().Bold());
  EXPECT_EQ(Render(buffer),
            "| plain\n"
            "  more bold\n"
            "  text\n");

  // Carriage returns return there too, and tab stops are counted from it.
  Buffer positional(20, Charset::Ascii);
  positional.DrawText(0, 0, "| ", Style());
  positional.DrawText(2, 0, 2, "a\tb\rc", Style());
  EXPECT_EQ(Render(positional), "| c       b\n");
}

TEST(BufferTest, TextControlCharacters) {
  Buffer buffer(10, Charset::Utf8);
  // A carriage return returns to the column the text started in.
  buffer.DrawText(2, 0, "abc\rx", Style());
  EXPECT_EQ(Render(buffer), "  xbc\n");

  // Anything else with no printable rendering is replaced, so a stray byte
  // can't shift the columns after it.
  Buffer utf8(10, Charset::Utf8);
  utf8.DrawText(0, 0,
                "a\x01"
                "b",
                Style());
  EXPECT_EQ(Render(utf8), "a�b\n");

  Buffer ascii(10, Charset::Ascii);
  ascii.DrawText(0, 0,
                 "a\x01"
                 "b",
                 Style());
  EXPECT_EQ(Render(ascii), "a?b\n");
}

TEST(BufferTest, AsciiDoesNoUtf8Processing) {
  // A terminal that isn't decoding UTF-8 renders each byte as some character
  // of its own, so every byte has to be counted as one column. Replacing the
  // bytes keeps the column count honest without guessing at an encoding.
  Buffer buffer(10, Charset::Ascii);
  buffer.DrawText(0, 0, "a中b", Style());
  EXPECT_EQ(Render(buffer), "a???b\n");
  EXPECT_EQ(buffer.MeasureText(0, 0, "a中b").x, 5);

  // The same goes for text with combining marks, which occupy no columns only
  // because a UTF-8 terminal folds them into the one before.
  Buffer marks(10, Charset::Ascii);
  marks.DrawText(0, 0, AcuteE, Style());
  EXPECT_EQ(Render(marks), "e??\n");

  // Invalid UTF-8 is not even a category here; bytes are bytes.
  Buffer invalid(10, Charset::Ascii);
  invalid.DrawText(0, 0, llvm::StringRef("\xc0\x80z", 3), Style());
  EXPECT_EQ(Render(invalid), "??z\n");

  // Every code point is one column wide, whatever it is.
  EXPECT_EQ(buffer.DrawCodePoint(0, 1, U'中', Style()).x, 1);
  EXPECT_EQ(buffer.DrawCodePoint(1, 1, CombiningAcute, Style()).x, 2);
  EXPECT_EQ(Render(buffer),
            "a???b\n"
            "??\n");
}

TEST(BufferTest, TextInvalidUtf8) {
  Buffer buffer(10, Charset::Utf8);
  // An overlong encoding of NUL. It is rejected, and decoding resynchronizes
  // one byte at a time rather than giving up on the rest of the text.
  buffer.DrawText(0, 0, llvm::StringRef("\xc0\x80z", 3), Style());
  EXPECT_EQ(Render(buffer), "��z\n");

  Buffer surrogate(10, Charset::Utf8);
  surrogate.DrawText(0, 0, llvm::StringRef("\xed\xa0\x80z", 4), Style());
  EXPECT_EQ(Render(surrogate), "���z\n");
}

TEST(BufferTest, CodePointsWithNoEncoding) {
  // Decoding text never yields these, but `DrawCodePoint` takes any code point,
  // including the surrogates and the values past the last one that UTF-8 has no
  // encoding for.
  Buffer buffer(10, Charset::Utf8);
  EXPECT_EQ(
      buffer.DrawCodePoint(0, 0, static_cast<char32_t>(0xd800), Style()).x, 1);
  EXPECT_EQ(
      buffer.DrawCodePoint(1, 0, static_cast<char32_t>(0xdfff), Style()).x, 2);
  EXPECT_EQ(
      buffer.DrawCodePoint(2, 0, static_cast<char32_t>(0x110000), Style()).x,
      3);
  EXPECT_EQ(Render(buffer), "���\n");
}

TEST(BufferTest, DoubleWidthCharacters) {
  Buffer buffer(6, Charset::Utf8);
  buffer.DrawText(0, 0, "中A🔥", Style());
  EXPECT_EQ(Render(buffer), "中A🔥\n");
  EXPECT_EQ(buffer.DrawCodePoint(0, 1, U'中', Style()).x, 2);
}

TEST(BufferTest, DrawingOverADoubleWidthCharacterErasesAllOfIt) {
  // Overwriting either half has to erase the whole character. Either half left
  // behind misaligns what follows, because the columns the grid counts for the
  // cell and the ones the terminal paints stop agreeing.
  Buffer over_head(4, Charset::Utf8);
  over_head.DrawCodePoint(0, 0, U'中', Style());
  over_head.DrawCodePoint(0, 0, 'A', Style());
  EXPECT_EQ(Render(over_head), "A\n");

  Buffer over_tail(4, Charset::Utf8);
  over_tail.DrawCodePoint(0, 0, U'中', Style());
  over_tail.DrawCodePoint(1, 0, 'B', Style());
  EXPECT_EQ(Render(over_tail), " B\n");

  // The same holds when a double-width character lands on another one.
  Buffer over_both(6, Charset::Utf8);
  over_both.DrawCodePoint(0, 0, U'中', Style());
  over_both.DrawCodePoint(2, 0, U'中', Style());
  over_both.DrawCodePoint(1, 0, U'国', Style());
  EXPECT_EQ(Render(over_both), " 国\n");
}

TEST(BufferTest, DoubleWidthCharacterInTheLastColumnTakesBothColumns) {
  // It starts inside the width, and splitting one would leave the terminal
  // rendering half a character, so it overhangs by a column rather than being
  // refused.
  Buffer buffer(3, Charset::Utf8);
  EXPECT_EQ(buffer.DrawCodePoint(2, 0, U'中', Style()).x, 4);
  buffer.DrawCodePoint(0, 0, 'a', Style());
  EXPECT_EQ(Render(buffer), "a 中\n");
  EXPECT_GE(buffer.width(), 4);
}

TEST(BufferTest, CombiningMarks) {
  // Marks render into the column before them, so a base character and its
  // marks stay in one cell and don't shift what follows.
  Buffer buffer(10, Charset::Utf8);
  buffer.DrawText(0, 0, AcuteE, Style());
  EXPECT_EQ(Render(buffer), AcuteE.str() + "\n");
  EXPECT_EQ(buffer.MeasureText(0, 0, AcuteE).x, 1);
  EXPECT_EQ(buffer.DrawCodePoint(5, 0, CombiningAcute, Style()).x, 5);

  // Drawing over the base takes its marks with it.
  Buffer overwritten(10, Charset::Utf8);
  overwritten.DrawText(0, 0, AcuteE, Style());
  overwritten.DrawText(1, 0, "x", Style());
  overwritten.DrawCodePoint(0, 0, 'o', Style());
  EXPECT_EQ(Render(overwritten), "ox\n");
}

TEST(BufferTest, CombiningMarksOnADoubleWidthBase) {
  // A mark following a double-width character arrives at the column past its
  // continuation, and has to reach back to the character itself.
  Buffer buffer(10, Charset::Utf8);
  buffer.DrawText(0, 0, ("中" + AcuteE.drop_front(1) + "x").str(), Style());
  EXPECT_EQ(Render(buffer), ("中" + AcuteE.drop_front(1) + "x\n").str());
  EXPECT_EQ(buffer.MeasureText(0, 0, ("中" + AcuteE.drop_front(1)).str()).x, 2);
}

TEST(BufferTest, CombiningMarksAreCapped) {
  // Marks stack without bound in adversarial text, and every one of them would
  // otherwise land in a single cell's output.
  std::string zalgo = "e";
  for (int i = 0; i < 100; ++i) {
    zalgo += AcuteE.drop_front(1);
  }

  Buffer buffer(10, Charset::Utf8);
  buffer.DrawText(0, 0, zalgo, Style());
  // The base, some bounded run of marks, and the newline.
  EXPECT_LT(Render(buffer).size(), 64U);
  EXPECT_GT(Render(buffer).size(), 1U);
}

TEST(BufferTest, WrappedText) {
  Buffer buffer(20, Charset::Ascii);
  EXPECT_EQ(buffer
                .DrawWrappedText(0, 0, 0, 10,
                                 "This is a long sentence that should be "
                                 "wrapped.",
                                 Style())
                .y,
            5);

  EXPECT_EQ(Render(buffer),
            "This is a\n"
            "long\n"
            "sentence\n"
            "that\n"
            "should be\n"
            "wrapped.\n");
}

TEST(BufferTest, WrappedTextKeepsWordsTooLongToFitWhole) {
  // A break is a newline in the rendered text, and one inside a word stops it
  // being copied out in one piece, so the word overhangs the width instead and
  // the buffer grows to hold it.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 5, "abcdefghij", Style()).y, 0);
  EXPECT_EQ(Render(buffer), "abcdefghij\n");
}

TEST(BufferTest, WrappedTextStartsAnOverlongWordOnItsOwnRow) {
  // A word too long for any row moves to one of its own anyway, so it overhangs
  // from the margin rather than from wherever the previous word ended.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 5, "ab abcdefghij", Style()).y, 1);
  EXPECT_EQ(Render(buffer),
            "ab\n"
            "abcdefghij\n");
}

TEST(BufferTest, WrappedSpansShareOneBlock) {
  // A block whose spans are styled differently is drawn one span at a time,
  // each continuing where the last ended and all naming the same margin, so
  // the rows after the first line up with the block rather than with wherever
  // the span happened to start.
  Buffer buffer(20, Charset::Ascii);
  Buffer::DrawEnd end =
      buffer.DrawWrappedText(2, 0, 2, 12, "plain words", Style());
  end = buffer.DrawWrappedText(end.x, end.y, 2, 12, " emphasized more",
                               Style().Bold());
  buffer.DrawWrappedText(end.x, end.y, 2, 12, " plain again", Style());

  EXPECT_EQ(Render(buffer),
            "  plain words\n"
            "  emphasized\n"
            "  more plain\n"
            "  again\n");
}

TEST(BufferTest, WrappedTextIndents) {
  // Wrapping is relative to the margin rather than to where the text starts,
  // which is what lets a wrapped block sit beside a gutter.
  Buffer buffer(12, Charset::Ascii);
  buffer.DrawText(0, 0, "| ", Style());
  buffer.DrawWrappedText(2, 0, 2, 6, "alpha beta gamma", Style());
  EXPECT_EQ(Render(buffer),
            "| alpha\n"
            "  beta\n"
            "  gamma\n");
}

TEST(BufferTest, WrappedTextExpandsTabsToStops) {
  // A tab advances to the next stop, and one following a newline the text wrote
  // is indentation and is kept.
  Buffer buffer(40, Charset::Ascii);
  buffer.DrawWrappedText(0, 0, 0, 30, "a\tb\nlonger\tc", Style());
  EXPECT_EQ(Render(buffer),
            "a       b\n"
            "longer  c\n");

  // A tab the text wrote after a newline is indentation, and is kept.
  Buffer indented(40, Charset::Ascii);
  indented.DrawWrappedText(0, 0, 0, 30, "a\n\tb", Style());
  EXPECT_EQ(Render(indented),
            "a\n"
            "        b\n");
}

TEST(BufferTest, WrappedTextTabStopsFollowTheMargin) {
  // Stops are counted from the block's margin rather than from the buffer's
  // left edge.
  Buffer buffer(40, Charset::Ascii);
  buffer.DrawText(0, 0, "| ", Style());
  buffer.DrawWrappedText(2, 0, 2, 20, "a\tb", Style());
  EXPECT_EQ(Render(buffer), "| a       b\n");
}

TEST(BufferTest, WrappedTextBreaksAtTabs) {
  // A tab is a break opportunity as well as a jump to a stop.
  Buffer buffer(40, Charset::Ascii);
  buffer.DrawWrappedText(0, 0, 0, 8, "aaaa\tbbbb", Style());
  EXPECT_EQ(Render(buffer),
            "aaaa\n"
            "bbbb\n");

  // One reaching past the block stops at its edge, so the span after it
  // continues from there rather than from a stop outside the block.
  Buffer clipped(40, Charset::Ascii);
  EXPECT_EQ(clipped.DrawWrappedText(0, 0, 0, 4, "ab\t", Style()),
            DrawEnd(4, 0));
}

TEST(BufferTest, TabWidthComesFromCapabilities) {
  Capabilities capabilities;
  capabilities.charset = Charset::Ascii;
  capabilities.tab_width = 4;

  Buffer buffer(capabilities);
  buffer.DrawText(0, 0, "a\tb", Style());
  buffer.DrawWrappedText(0, 1, 0, 20, "a\tb", Style());
  EXPECT_EQ(Render(buffer),
            "a   b\n"
            "a   b\n");

  // A terminal claiming stops a buffer can't draw to is clamped rather than
  // trusted, the same as one claiming an impossible width.
  capabilities.tab_width = 0;
  Buffer clamped(capabilities);
  clamped.DrawText(0, 0, "a\tb", Style());
  EXPECT_EQ(Render(clamped), "a b\n");
}

TEST(BufferTest, WrappedTextKeepsTheBreaksItIsGiven) {
  // Wrapping only adds breaks. Text that arrives wrapped to some other width
  // keeps that wrapping rather than being reflowed into this one, even where
  // its lines would fit together.
  Buffer buffer(40, Charset::Ascii);
  EXPECT_EQ(
      buffer.DrawWrappedText(0, 0, 0, 30, "already\nwrapped\nnarrow", Style())
          .y,
      2);
  EXPECT_EQ(Render(buffer),
            "already\n"
            "wrapped\n"
            "narrow\n");

  // A break the text ends with closes its last line, and a row nothing was
  // drawn into is not a row, so it doesn't also open an empty one.
  Buffer trailing(40, Charset::Ascii);
  trailing.DrawWrappedText(0, 0, 0, 30, "one\ntwo\n", Style());
  EXPECT_EQ(Render(trailing),
            "one\n"
            "two\n");
}

TEST(BufferTest, WrappedTextLineBreaks) {
  // Carriage returns are dropped, so CRLF endings break exactly once.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 10, "a\r\nb", Style()).y, 1);
  EXPECT_EQ(Render(buffer),
            "a\n"
            "b\n");

  // Whitespace stops at the block's edge, so a wrapped row starts at the
  // margin.
  Buffer spaces(10, Charset::Ascii);
  spaces.DrawWrappedText(0, 0, 0, 5, "aaaaa     bbbbb", Style());
  EXPECT_EQ(Render(spaces),
            "aaaaa\n"
            "bbbbb\n");
}

TEST(BufferTest, WrappedTextFillsTheWidthLeftOfTheMargin) {
  // A caller with nothing to divide the width between gives the block all of
  // what is left of it, which is what wrapping to the terminal is.
  Buffer buffer(20, Charset::Ascii);
  buffer.DrawText(0, 0, "-> ", Style());
  buffer.DrawWrappedText(3, 0, 3, buffer.columns() - 3,
                         "several words that would otherwise fit", Style());
  EXPECT_EQ(Render(buffer),
            "-> several words\n"
            "   that would\n"
            "   otherwise fit\n");
}

TEST(BufferTest, WrappedTextKeepsAMarkWithTheWhitespaceBeforeIt) {
  // A combining mark renders into the column before it, which for a mark
  // following whitespace is that whitespace. Left to begin the word after it,
  // the mark would move to another row whenever that word wrapped and attach to
  // whatever preceded it there.
  Buffer buffer(10, Charset::Utf8);
  std::string mark = AcuteE.drop_front(1).str();
  buffer.DrawWrappedText(0, 0, 0, 5, "aaa " + mark + "bbbb", Style());
  EXPECT_EQ(Render(buffer), "aaa " + mark + "\nbbbb\n");

  // Whitespace stops at the block's edge, so a mark can arrive where none of it
  // was drawn. It attaches to the last cell written rather than being carried
  // to the row the next word wraps onto.
  Buffer filled(10, Charset::Utf8);
  filled.DrawWrappedText(0, 0, 0, 5, "aaaaa " + mark + "bb", Style());
  EXPECT_EQ(Render(filled), "aaaaa" + mark + "\nbb\n");
}

TEST(BufferTest, WrappedTextKeepsCharactersWiderThanTheRegion) {
  // No row in a one-column region could hold a double-width character. Drawing
  // it anyway overruns the region, which is what keeping the text costs here.
  Buffer buffer(10, Charset::Utf8);
  buffer.DrawWrappedText(0, 0, 0, 1, "中中", Style());
  EXPECT_EQ(Render(buffer), "中中\n");
}

TEST(BufferTest, WrappedTextWithDoubleWidthCharacters) {
  // Wrapping counts columns, not characters, so half as many double-width ones
  // fit a row.
  Buffer buffer(10, Charset::Utf8);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 4, "中中 中中", Style()).y, 1);
  EXPECT_EQ(Render(buffer),
            "中中\n"
            "中中\n");
}

TEST(BufferTest, TrailingBlanksAreDropped) {
  // Padding out to the buffer's width would put invisible whitespace into
  // every line of output, which shows up in diffs and in copied text.
  Buffer buffer(40, Charset::Ascii);
  buffer.DrawText(0, 0, "hi", Style());
  EXPECT_EQ(Render(buffer), "hi\n");
}

TEST(BufferTest, StyledBlanksAreKept) {
  // A blank cell with a background still paints, so it isn't padding.
  Buffer buffer(10, Charset::Ascii);
  buffer.DrawCodePoint(0, 0, 'x', Style());
  buffer.DrawCodePoint(3, 0, ' ', Style().Background(AnsiColor::Red));
  EXPECT_EQ(Render(buffer, ColorMode::Ansi16), "x  \x1b[41m \x1b[0m\n");

  // With color off the background paints nothing, so those cells are padding
  // again and must not reach the output as trailing spaces.
  EXPECT_EQ(Render(buffer, ColorMode::NoColor), "x\n");
}

TEST(BufferTest, RenderMinimizesEscapes) {
  Buffer buffer(4, Charset::Ascii);
  Style red_bold = Style().Bold().Foreground(Color(255, 0, 0));
  buffer.DrawCodePoint(0, 0, 'A', red_bold);
  // Sharing the attributes and changing only the color costs one escape.
  buffer.DrawCodePoint(1, 0, 'B', red_bold.Foreground(Color(0, 0, 255)));
  // Dropping bold costs a reset and a fresh start.
  buffer.DrawCodePoint(2, 0, 'C', Style().Foreground(Color(0, 0, 255)));
  buffer.DrawCodePoint(3, 0, 'D', Style());

  EXPECT_EQ(Render(buffer, ColorMode::Truecolor),
            "\x1b[1m\x1b[38;2;255;0;0m"
            "A"
            "\x1b[38;2;0;0;255m"
            "B"
            "\x1b[0m\x1b[38;2;0;0;255m"
            "C"
            "\x1b[0m"
            "D\n");
}

TEST(BufferTest, MeasureText) {
  Buffer utf8(10, Charset::Utf8);
  EXPECT_EQ(utf8.MeasureText(0, 0, ""), DrawEnd(0, 0));
  EXPECT_EQ(utf8.MeasureText(0, 0, "hello"), DrawEnd(5, 0));
  EXPECT_EQ(utf8.MeasureText(0, 0, "中A🔥"), DrawEnd(5, 0));
  EXPECT_EQ(utf8.MeasureText(0, 0, AcuteE), DrawEnd(1, 0));
  EXPECT_EQ(utf8.MeasureText(0, 0, llvm::StringRef("\xc0\x80", 2)),
            DrawEnd(2, 0));

  // Newlines, carriage returns, and tabs are interpreted as drawing does.
  EXPECT_EQ(utf8.MeasureText(0, 0, "a\nb"), DrawEnd(1, 1));
  EXPECT_EQ(utf8.MeasureText(0, 0, "a\r\nb"), DrawEnd(1, 1));
  EXPECT_EQ(utf8.MeasureText(0, 0, "a\tb"), DrawEnd(9, 0));

  // Measuring starts from where it is told to, which is the column a newline
  // returns to and the origin the tab stops count from.
  EXPECT_EQ(utf8.MeasureText(3, 2, "a\tb"), DrawEnd(12, 2));
  EXPECT_EQ(utf8.MeasureText(3, 2, "a\nb"), DrawEnd(4, 3));

  // Every byte is a column when the terminal isn't decoding UTF-8.
  Buffer ascii(10, Charset::Ascii);
  EXPECT_EQ(ascii.MeasureText(0, 0, "hello"), DrawEnd(5, 0));
  EXPECT_EQ(ascii.MeasureText(0, 0, "中A🔥"), DrawEnd(8, 0));
  EXPECT_EQ(ascii.MeasureText(0, 0, AcuteE), DrawEnd(3, 0));
}

TEST(BufferTest, MeasuringMatchesDrawing) {
  // The point of measuring is to answer what drawing would, so check the two
  // against each other on the text most likely to make them disagree.
  for (llvm::StringRef text :
       {"hello", "a\tb\tc", "a\nb\r\nc", "中A🔥", "a  b", ""}) {
    Buffer buffer(10, Charset::Utf8);
    EXPECT_EQ(buffer.MeasureText(2, 1, text),
              buffer.DrawText(2, 1, text, Style()))
        << text;

    // A margin left of where the text starts moves what the positional
    // characters answer to, and moves it for both of them alike.
    Buffer block(10, Charset::Utf8);
    EXPECT_EQ(block.MeasureText(2, 1, 1, text),
              block.DrawText(2, 1, 1, text, Style()))
        << text;
  }
  for (llvm::StringRef text :
       {"one two three", "a\nlonger line here", "verylongunbreakableword",
        "a\tb\tc", "col\tone\nrow\ttwo", "e\xcc\x81 \xcc\x81word", ""}) {
    Buffer buffer(10, Charset::Utf8);
    EXPECT_EQ(buffer.MeasureWrappedText(2, 1, 2, 8, text),
              buffer.DrawWrappedText(2, 1, 2, 8, text, Style()))
        << text;
  }
}

TEST(BufferTest, MeasureWrapWidth) {
  Buffer buffer(10, Charset::Utf8);
  EXPECT_EQ(buffer.MeasureWrapWidth(""), 0);
  EXPECT_EQ(buffer.MeasureWrapWidth("a bb ccc"), 3);
  EXPECT_EQ(buffer.MeasureWrapWidth("  spaced  out  "), 6);

  // Newlines and tabs bound a word without taking columns of their own.
  EXPECT_EQ(buffer.MeasureWrapWidth("a\nbb\tccc"), 3);

  // A word is measured in the columns it takes, not the bytes it holds.
  EXPECT_EQ(buffer.MeasureWrapWidth("中中 a"), 4);
}

TEST(BufferTest, WrapWidthIsWhatWrappingDoesNotOverhang) {
  // The width answered for is a fact about wrapping, so it is checked against
  // the wrapping it describes.
  Buffer buffer(10, Charset::Utf8);
  // Tabs don't widen the answer: one stops at the block's edge rather than
  // running to a stop outside it, so it can't overhang either.
  for (llvm::StringRef text :
       {"some quite long words here", "some\tquite\tlong words here"}) {
    int width = buffer.MeasureWrapWidth(text);
    Buffer drawn(width, Charset::Utf8);
    drawn.DrawWrappedText(0, 0, 0, width, text, Style());
    llvm::SmallVector<llvm::StringRef> rows;
    llvm::StringRef(Render(drawn)).split(rows, '\n');
    for (llvm::StringRef row : rows) {
      EXPECT_LE(static_cast<int>(row.size()), width) << row;
    }
  }
}

TEST(BufferTest, BuiltFromCapabilities) {
  Capabilities capabilities;
  capabilities.columns = 5;
  capabilities.charset = Charset::Utf8;

  Buffer buffer(capabilities);
  EXPECT_EQ(buffer.columns(), 5);
  EXPECT_EQ(buffer.charset(), Charset::Utf8);
  buffer.DrawHorizontalLine(0, 0, 5, Style());
  EXPECT_EQ(Render(buffer, capabilities.color_mode), "╶───╴\n");

  // A terminal that said nothing about its width gets one chosen to be safe
  // wherever the output ends up, rather than no width at all.
  capabilities.columns = std::nullopt;
  Buffer fallback(capabilities);
  EXPECT_EQ(fallback.columns(), DefaultColumns);

  // One claiming a width no grid can hold gets the nearest that can be held.
  capabilities.columns = Buffer::MaxColumns + 1;
  Buffer clamped(capabilities);
  EXPECT_EQ(clamped.columns(), Buffer::MaxColumns);
}

TEST(BufferTest, CombiningMarkPastTheWidthItStartedWith) {
  // The buffer grows to hold the text, so a mark arriving past the width it
  // was constructed with attaches to its base like any other.
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawText(0, 0, ("abcde" + AcuteE.drop_front(1)).str(), Style());
  EXPECT_EQ(Render(buffer), ("abcde" + AcuteE.drop_front(1) + "\n").str());
}

TEST(BufferTest, CombiningMarksSurviveGrowthAndOverdraw) {
  // Marks live in a side table keyed by cell index, so widening has to move
  // them with the rows and overdrawing has to take them with the cell.
  for (int start : {1, 2, 3}) {
    Buffer buffer(start, Charset::Utf8);
    buffer.DrawText(0, 0, std::string(AcuteE) + "e" + std::string(AcuteE),
                    Style());
    buffer.DrawText(0, 1, "xxxxxxxxxxxx", Style());
    buffer.DrawCodePoint(0, 0, U'z', Style());
    std::string rendered = Render(buffer);
    // The mark on the overdrawn cell went with it; the later one stayed.
    EXPECT_EQ(rendered.substr(0, rendered.find('\n')),
              "ze" + std::string(AcuteE))
        << start;
  }
}

TEST(BufferTest, CombiningMarkWithNoBase) {
  // Marks render into the column before them, so one at the start of a row has
  // nowhere to go rather than attaching to the end of the row above. This is
  // ordinary input rather than a caller mistake -- a source file can open a
  // line with a mark -- so it is dropped and drawing goes on.
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawText(0, 0, "ab", Style());
  buffer.DrawText(0, 1, AcuteE.drop_front(1).str(), Style());
  EXPECT_EQ(Render(buffer), "ab\n");

  // Nor does it disturb what is already drawn on the row it lands on.
  Buffer after(4, Charset::Utf8);
  after.DrawText(0, 0, "ab", Style());
  after.DrawCodePoint(0, 0, CombiningAcute, Style());
  EXPECT_EQ(Render(after), "ab\n");
}

TEST(BufferTest, OverhangStopsAtTheGridBound) {
  // An overhanging word is the only thing that reaches the grid's far edge, and
  // how far it reaches is a fact about the text rather than something a caller
  // could have checked, so it is clipped there rather than being a mistake.
  // The column still advances past it, which is what keeps measuring and
  // drawing answering the same thing.
  Buffer buffer(4, Charset::Ascii);
  std::string word(Buffer::MaxColumns + 10, 'x');
  DrawEnd past(Buffer::MaxColumns + 10, 0);
  EXPECT_EQ(buffer.MeasureWrappedText(0, 0, 0, 4, word), past);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 4, word, Style()), past);
  EXPECT_EQ(buffer.width(), Buffer::MaxColumns);
  EXPECT_EQ(buffer.height(), 1);
}

TEST(BufferTest, ACharacterStraddlingTheGridBoundIsNotDrawn) {
  // Whether a character fits the far edge depends on how wide it turns out to
  // be, and half of one is not something a terminal can render, so one
  // straddling the bound is past it entirely.
  Buffer buffer(4, Charset::Utf8);
  std::string word(Buffer::MaxColumns - 1, 'x');
  word += "中";
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 0, 4, word, Style()),
            DrawEnd(Buffer::MaxColumns + 1, 0));
  EXPECT_EQ(buffer.width(), Buffer::MaxColumns);
  // Only the run before it reached the output; the character itself did not.
  EXPECT_EQ(Render(buffer).size(), static_cast<size_t>(Buffer::MaxColumns));
}

TEST(BufferTest, APointBoxLeavesALineAlone) {
  // A point is line art with no directions, so drawing one where a line
  // already runs adds nothing and leaves the line's own directions in place.
  Buffer buffer(3, Charset::Utf8);
  buffer.DrawHorizontalLine(0, 0, 3, Style());
  buffer.DrawBox(1, 0, 1, 1, Style());
  EXPECT_EQ(Render(buffer), "╶─╴\n");

  // With nothing there, the point is drawn.
  Buffer empty(3, Charset::Utf8);
  empty.DrawBox(1, 0, 1, 1, Style());
  EXPECT_EQ(Render(empty), " ·\n");
}

TEST(BufferDeathTest, WidthMustFitTheGrid) {
  // A width comes from `COLUMNS` by way of `Capabilities`, which clamps it. A
  // caller reaching this constructor has computed the width itself, so a value
  // the grid can't hold is a mistake rather than bad input.
  EXPECT_DEATH(Buffer(0, Charset::Ascii), "Buffer width must be in");
  EXPECT_DEATH(Buffer(-1, Charset::Ascii), "Buffer width must be in");
  EXPECT_DEATH(Buffer(Buffer::MaxColumns + 1, Charset::Ascii),
               "Buffer width must be in");
}

TEST(BufferDeathTest, TextMustStartInsideTheGrid) {
  // The width does not bind unwrapped text, so only a column before the origin
  // or a row no grid can index is a mistake.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_DEATH(buffer.DrawCodePoint(-1, 0, 'a', Style()), "is outside the");
  EXPECT_DEATH(buffer.DrawCodePoint(0, -1, 'a', Style()), "is outside the");
  EXPECT_DEATH(buffer.DrawCodePoint(0, Buffer::MaxRows, 'a', Style()),
               "is outside the");
  EXPECT_DEATH(buffer.DrawText(-1, 0, "a", Style()), "is outside the");
  EXPECT_DEATH(buffer.MeasureText(0, Buffer::MaxRows, "a"), "is outside the");

  // Text begins at or right of the margin its rows return to.
  EXPECT_DEATH(buffer.DrawText(2, 0, 3, "a", Style()), "left of its margin");
  EXPECT_DEATH(buffer.MeasureText(2, 0, -1, "a"), "left of its margin");
}

TEST(BufferTest, UnwrappedTextWidensTheBuffer) {
  // `DrawText` widens the buffer rather than being held to its width, so a run
  // continues from where the one before it ended however far past that is.
  Buffer buffer(10, Charset::Ascii);
  Buffer::DrawEnd end =
      buffer.DrawText(0, 0, "a message far longer than ten columns", Style());
  EXPECT_EQ(end, DrawEnd(37, 0));
  EXPECT_EQ(buffer.columns(), 10);
  EXPECT_GE(buffer.width(), 37);

  // Which is what lets a row be built from runs that carry different styles.
  buffer.DrawText(end.x, end.y, " [tag]", Style().Bold());
  EXPECT_EQ(Render(buffer), "a message far longer than ten columns [tag]\n");
}

TEST(BufferDeathTest, LinesMustFitWhatTheyAreDrawnInto) {
  // Nothing about a line is unbreakable, so unlike text it has no reason to
  // reach outside the width, and one that does came from a wrong extent.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_DEATH(buffer.DrawHorizontalLine(6, 0, 5, Style()), "runs outside the");
  EXPECT_DEATH(buffer.DrawHorizontalLine(0, 0, -1, Style()),
               "runs outside the");
  EXPECT_DEATH(buffer.DrawVerticalLine(0, 0, Buffer::MaxRows + 1, Style()),
               "runs outside the");
  EXPECT_DEATH(buffer.DrawBox(0, 0, 11, 2, Style()), "runs outside the");
}

TEST(BufferDeathTest, WrappedBlocksMustFitTheWidth) {
  // A block is a division of the width rather than something that can exceed
  // it, and the text has to start inside the block it wraps in.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_DEATH(buffer.DrawWrappedText(0, 0, 0, 11, "a", Style()),
               "does not fit the");
  EXPECT_DEATH(buffer.DrawWrappedText(4, 0, 4, 8, "a", Style()),
               "does not fit the");
  EXPECT_DEATH(buffer.DrawWrappedText(0, 0, 0, 0, "a", Style()),
               "does not fit the");
  EXPECT_DEATH(buffer.DrawWrappedText(2, 0, 4, 6, "a", Style()),
               "does not fit the");
  EXPECT_DEATH(buffer.MeasureWrappedText(0, 0, 0, 11, "a"), "does not fit the");
}

TEST(BufferDeathTest, TextMustBeShortEnoughToMeasure) {
  // Measuring walks the text adding widths, so a long enough run would carry
  // the column past what an `int` holds. Text this long was built rather than
  // read off a line.
  Buffer buffer(10, Charset::Ascii);
  std::string huge(Buffer::MaxTextBytes + 1, 'x');
  EXPECT_DEATH(buffer.DrawText(0, 0, huge, Style()), "is past the");
  EXPECT_DEATH(buffer.MeasureText(0, 0, huge), "is past the");
}

TEST(BufferTest, WriteTo) {
  Buffer buffer(20, Charset::Ascii);
  buffer.DrawText(0, 0, "hello", Style().Bold());
  buffer.DrawText(0, 1, "world", Style());

  auto dir = Filesystem::MakeTmpDir();
  ASSERT_TRUE(dir.ok()) << dir.error();
  auto file = dir->OpenWriteOnly("out", Filesystem::CreationOptions::CreateNew);
  ASSERT_TRUE(file.ok()) << file.error();
  auto written = buffer.WriteTo(*file, ColorMode::Ansi16);
  EXPECT_TRUE(written.ok()) << written.error();
  (*std::move(file)).Close().Check();

  // Byte for byte what `Render` produces.
  auto read_back = dir->ReadFileToString("out");
  ASSERT_TRUE(read_back.ok()) << read_back.error();
  EXPECT_EQ(*read_back, Render(buffer, ColorMode::Ansi16));
  EXPECT_EQ(*read_back, "\x1b[1mhello\n\x1b[0mworld\n");
}

TEST(BufferTest, StyleCarriesAcrossRows) {
  // A style is usually still in use on the row below, so turning it off at the
  // end of a row and back on at the start of the next costs a reset and a fresh
  // start for nothing.
  Buffer buffer(4, Charset::Ascii);
  buffer.DrawText(0, 0, "ab", Style().Bold());
  buffer.DrawText(0, 1, "cd", Style().Bold());
  EXPECT_EQ(Render(buffer, ColorMode::Ansi16), "\x1b[1mab\ncd\x1b[0m\n");
}

TEST(BufferTest, StyleThatPaintsBlanksStopsAtTheRowEnd) {
  // A terminal fills the rest of a row with the background it is in when the
  // row ends, so a style that paints where there is no glyph can't be left on
  // across the newline the way an attribute that only affects a glyph can.
  Buffer buffer(4, Charset::Ascii);
  buffer.DrawText(0, 0, "ab", Style().Background(AnsiColor::Red));
  buffer.DrawText(0, 1, "cd", Style().Background(AnsiColor::Red));
  EXPECT_EQ(Render(buffer, ColorMode::Ansi16),
            "\x1b[41mab\x1b[0m\n"
            "\x1b[41mcd\x1b[0m\n");
}

TEST(BufferTest, RenderAppends) {
  // Rendering appends, so several buffers can be gathered into one write.
  Buffer first(10, Charset::Ascii);
  first.DrawText(0, 0, "one", Style());
  Buffer second(10, Charset::Ascii);
  second.DrawText(0, 0, "two", Style());

  llvm::SmallString<64> out;
  first.Render(out, ColorMode::NoColor);
  second.Render(out, ColorMode::NoColor);
  EXPECT_EQ(std::string(out), "one\ntwo\n");
}

}  // namespace
}  // namespace Carbon::Terminal

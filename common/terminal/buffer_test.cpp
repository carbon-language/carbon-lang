// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>

#include "common/filesystem.h"
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
  EXPECT_EQ(buffer.DrawSymbol(0, 2, 'x', Style()).x, 1);
  EXPECT_EQ(buffer.height(), 3);
  EXPECT_EQ(Render(buffer), "\n\nx\n");
}

TEST(BufferTest, GrowsColumnsToFitWhatIsDrawn) {
  Buffer buffer(3, Charset::Ascii);
  buffer.DrawHorizontalLine(1, 0, 5, Style());
  buffer.DrawVerticalLine(0, 1, 5, Style());
  EXPECT_EQ(Render(buffer),
            " -----\n"
            "|\n"
            "|\n"
            "|\n"
            "|\n"
            "|\n");
  EXPECT_GE(buffer.width(), 6);
}

TEST(BufferTest, DrawsNothingBeforeItsOrigin) {
  Buffer buffer(3, Charset::Ascii);
  // Negative coordinates have nowhere to grow to. The symbols still report the
  // columns they would have taken, so text walking a row stays in step.
  EXPECT_EQ(buffer.DrawSymbol(-1, 0, 'a', Style()).x, 0);
  EXPECT_EQ(buffer.DrawSymbol(0, -1, 'a', Style()).x, 1);
  EXPECT_EQ(buffer.height(), 0);
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

TEST(BufferTest, LinesJoinWhereTheyMeet) {
  Buffer buffer(3, Charset::Utf8);
  buffer.DrawHorizontalLine(0, 1, 3, Style());
  buffer.DrawVerticalLine(1, 0, 3, Style());

  EXPECT_EQ(Render(buffer),
            " │\n"
            "─┼─\n"
            " │\n");
}

TEST(BufferTest, LineEndsFormCorners) {
  Buffer buffer(3, Charset::Utf8);
  buffer.DrawHorizontalLine(1, 0, 2, Style());
  buffer.DrawVerticalLine(1, 0, 3, Style());

  EXPECT_EQ(Render(buffer),
            " ╭─\n"
            " │\n"
            " │\n");
}

TEST(BufferTest, LinesOnlyJoinWhereTheyOverlap) {
  // A cell's glyph follows from the directions lines leave it in, so joining
  // is a matter of drawing into the same cell rather than of being adjacent.
  // This keeps drawing order from mattering and keeps text that looks like a
  // line from being redrawn as one.
  Buffer separate(3, Charset::Utf8);
  separate.DrawHorizontalLine(0, 0, 3, Style());
  separate.DrawVerticalLine(0, 1, 2, Style());
  EXPECT_EQ(Render(separate),
            "───\n"
            "│\n"
            "│\n");

  Buffer overlapping(3, Charset::Utf8);
  overlapping.DrawHorizontalLine(0, 0, 3, Style());
  overlapping.DrawVerticalLine(0, 0, 3, Style());
  EXPECT_EQ(Render(overlapping),
            "╭──\n"
            "│\n"
            "│\n");
}

TEST(BufferTest, DrawOrderDoesNotMatter) {
  Buffer vertical_first(3, Charset::Utf8);
  vertical_first.DrawVerticalLine(1, 0, 3, Style());
  vertical_first.DrawHorizontalLine(0, 1, 3, Style());

  EXPECT_EQ(Render(vertical_first),
            " │\n"
            "─┼─\n"
            " │\n");
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
}

TEST(BufferTest, SingleCellLines) {
  Buffer horizontal(3, Charset::Utf8);
  horizontal.DrawHorizontalLine(1, 0, 1, Style());
  EXPECT_EQ(Render(horizontal), " ─\n");

  Buffer vertical(3, Charset::Utf8);
  vertical.DrawVerticalLine(1, 0, 1, Style());
  EXPECT_EQ(Render(vertical), " │\n");

  // A line with no length draws nothing at all.
  Buffer empty(3, Charset::Utf8);
  empty.DrawHorizontalLine(0, 0, 0, Style());
  empty.DrawVerticalLine(0, 0, -1, Style());
  EXPECT_EQ(Render(empty), "");
}

TEST(BufferTest, LinesDrawOverContent) {
  // A line replaces whatever text was in the cell, including both halves of a
  // double-width symbol it lands on.
  Buffer over_text(5, Charset::Utf8);
  over_text.DrawText(0, 0, "abcde", Style());
  over_text.DrawHorizontalLine(1, 0, 3, Style());
  EXPECT_EQ(Render(over_text), "a───e\n");

  Buffer over_wide(5, Charset::Utf8);
  over_wide.DrawText(0, 0, "中中", Style());
  over_wide.DrawVerticalLine(1, 0, 1, Style());
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

  // ASCII can only tell horizontal and vertical apart from everything else.
  Buffer ascii(4, Charset::Ascii);
  ascii.DrawBox(0, 0, 4, 3, Style());
  EXPECT_EQ(Render(ascii),
            "+--+\n"
            "|  |\n"
            "+--+\n");

  // A box with no interior is the single line that bounds it.
  Buffer flat(4, Charset::Utf8);
  flat.DrawBox(0, 0, 4, 1, Style());
  flat.DrawBox(0, 2, 1, 2, Style());
  EXPECT_EQ(Render(flat),
            "────\n"
            "\n"
            "│\n"
            "│\n");

  Buffer degenerate(4, Charset::Utf8);
  degenerate.DrawBox(0, 0, 0, 4, Style());
  degenerate.DrawBox(0, 0, 4, -1, Style());
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

  // Empty text still occupies the row it started on.
  EXPECT_EQ(buffer.DrawText(0, 0, "", Style()).y, 0);
}

TEST(BufferTest, TabStopsAreMeasuredFromWhereTextBegins) {
  // Tab stops follow the text's own origin, not the left edge, so a source
  // line quoted beside a gutter keeps the tab alignment it had in the file.
  Buffer buffer(20, Charset::Utf8);
  buffer.DrawText(3, 0, "A\tB", Style());
  EXPECT_EQ(Render(buffer), "   A       B\n");
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

  // Every symbol is one column wide, whatever it is.
  EXPECT_EQ(buffer.DrawSymbol(0, 1, U'中', Style()).x, 1);
  EXPECT_EQ(buffer.DrawSymbol(1, 1, CombiningAcute, Style()).x, 2);
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

TEST(BufferTest, SymbolsWithNoEncoding) {
  // Decoding text never yields these, but `DrawSymbol` takes any code point,
  // including the surrogates and the values past the last one that UTF-8 has no
  // encoding for.
  Buffer buffer(10, Charset::Utf8);
  EXPECT_EQ(buffer.DrawSymbol(0, 0, static_cast<char32_t>(0xd800), Style()).x,
            1);
  EXPECT_EQ(buffer.DrawSymbol(1, 0, static_cast<char32_t>(0xdfff), Style()).x,
            2);
  EXPECT_EQ(buffer.DrawSymbol(2, 0, static_cast<char32_t>(0x110000), Style()).x,
            3);
  EXPECT_EQ(Render(buffer), "���\n");
}

TEST(BufferTest, DoubleWidthSymbols) {
  Buffer buffer(6, Charset::Utf8);
  buffer.DrawText(0, 0, "中A🔥", Style());
  EXPECT_EQ(Render(buffer), "中A🔥\n");
  EXPECT_EQ(buffer.DrawSymbol(0, 1, U'中', Style()).x, 2);
}

TEST(BufferTest, DrawingOverADoubleWidthSymbolErasesAllOfIt) {
  // Overwriting either half has to erase the whole symbol. Leaving the other
  // half behind would either duplicate part of a glyph or, for the trailing
  // half, silently drop a column and misalign everything after it.
  Buffer over_head(4, Charset::Utf8);
  over_head.DrawSymbol(0, 0, U'中', Style());
  over_head.DrawSymbol(0, 0, 'A', Style());
  EXPECT_EQ(Render(over_head), "A\n");

  Buffer over_tail(4, Charset::Utf8);
  over_tail.DrawSymbol(0, 0, U'中', Style());
  over_tail.DrawSymbol(1, 0, 'B', Style());
  EXPECT_EQ(Render(over_tail), " B\n");

  // The same holds when a double-width symbol lands on another one.
  Buffer over_both(6, Charset::Utf8);
  over_both.DrawSymbol(0, 0, U'中', Style());
  over_both.DrawSymbol(2, 0, U'中', Style());
  over_both.DrawSymbol(1, 0, U'国', Style());
  EXPECT_EQ(Render(over_both), " 国\n");
}

TEST(BufferTest, DoubleWidthSymbolPastTheEdgeTakesBothColumns) {
  // Splitting one would leave the terminal rendering half a character, so the
  // buffer grows by both of its columns rather than one.
  Buffer buffer(3, Charset::Utf8);
  EXPECT_EQ(buffer.DrawSymbol(2, 0, U'中', Style()).x, 4);
  buffer.DrawSymbol(0, 0, 'a', Style());
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
  EXPECT_EQ(buffer.DrawSymbol(5, 0, CombiningAcute, Style()).x, 5);

  // A mark with nothing before it has nothing to attach to.
  Buffer orphan(10, Charset::Utf8);
  orphan.DrawSymbol(0, 0, CombiningAcute, Style());
  EXPECT_EQ(Render(orphan), "");

  // Drawing over the base takes its marks with it.
  Buffer overwritten(10, Charset::Utf8);
  overwritten.DrawText(0, 0, AcuteE, Style());
  overwritten.DrawText(1, 0, "x", Style());
  overwritten.DrawSymbol(0, 0, 'o', Style());
  EXPECT_EQ(Render(overwritten), "ox\n");
}

TEST(BufferTest, CombiningMarksOnADoubleWidthBase) {
  // A mark following a double-width symbol arrives at the column past its
  // continuation, and has to reach back to the symbol itself.
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
                .DrawWrappedText(0, 0, 10,
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
  // Breaking one costs a reader more than the overhang does, so it overhangs
  // and the buffer grows to hold it.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 5, "abcdefghij", Style()).y, 0);
  EXPECT_EQ(Render(buffer), "abcdefghij\n");
}

TEST(BufferTest, WrappedTextStartsAnOverlongWordOnItsOwnRow) {
  // It still moves down to a row of its own, so the words before it aren't
  // pushed out of the region with it.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 5, "ab abcdefghij", Style()).y, 1);
  EXPECT_EQ(Render(buffer),
            "ab\n"
            "abcdefghij\n");
}

TEST(BufferTest, WrappedTextIndents) {
  // Wrapping is relative to where the text starts, which is what lets a
  // wrapped block sit beside a gutter.
  Buffer buffer(12, Charset::Ascii);
  buffer.DrawText(0, 0, "| ", Style());
  buffer.DrawWrappedText(2, 0, 6, "alpha beta gamma", Style());
  EXPECT_EQ(Render(buffer),
            "| alpha\n"
            "  beta\n"
            "  gamma\n");
}

TEST(BufferTest, WrappedTextLineBreaks) {
  // Carriage returns are dropped, so CRLF endings break exactly once.
  Buffer buffer(10, Charset::Ascii);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 10, "a\r\nb", Style()).y, 1);
  EXPECT_EQ(Render(buffer),
            "a\n"
            "b\n");

  // Spaces that would begin a row are dropped, keeping text aligned.
  Buffer spaces(10, Charset::Ascii);
  spaces.DrawWrappedText(0, 0, 5, "aaaaa     bbbbb", Style());
  EXPECT_EQ(Render(spaces),
            "aaaaa\n"
            "bbbbb\n");

  // A wrap region with no columns has nowhere to put anything.
  Buffer degenerate(10, Charset::Ascii);
  EXPECT_EQ(degenerate.DrawWrappedText(0, 0, 0, "anything", Style()).y, 0);
  EXPECT_EQ(degenerate.DrawWrappedText(0, 0, -1, "anything", Style()).y, 0);
  EXPECT_EQ(Render(degenerate), "");
}

TEST(BufferTest, WrappedTextWithNoWidthToFit) {
  // A caller with no width to fit passes one no row reaches, which has to wrap
  // nothing rather than overflow the column it would name.
  Buffer buffer(10, Charset::Ascii);
  llvm::StringRef text = "several words that would otherwise wrap";
  EXPECT_EQ(buffer.DrawWrappedText(3, 0, std::numeric_limits<int>::max(), text,
                                   Style()),
            DrawEnd(3 + static_cast<int>(text.size()), 0));
  EXPECT_EQ(Render(buffer), "   several words that would otherwise wrap\n");
}

TEST(BufferTest, WrappedTextKeepsSymbolsWiderThanTheRegion) {
  // No row in a one-column region could hold a double-width symbol. Drawing it
  // anyway overruns the region, which is what keeping the text costs here.
  Buffer buffer(10, Charset::Utf8);
  buffer.DrawWrappedText(0, 0, 1, "中中", Style());
  EXPECT_EQ(Render(buffer), "中中\n");
}

TEST(BufferTest, WrappedTextWithDoubleWidthSymbols) {
  // Wrapping counts columns, not characters, so half as many double-width ones
  // fit a row.
  Buffer buffer(10, Charset::Utf8);
  EXPECT_EQ(buffer.DrawWrappedText(0, 0, 4, "中中 中中", Style()).y, 1);
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
  buffer.DrawSymbol(0, 0, 'x', Style());
  buffer.DrawSymbol(3, 0, ' ', Style().Background(AnsiColor::Red));
  EXPECT_EQ(Render(buffer, ColorMode::Ansi16), "x  \x1b[41m \x1b[0m\n");

  // With color off the background paints nothing, so those cells are padding
  // again and must not reach the output as trailing spaces.
  EXPECT_EQ(Render(buffer, ColorMode::NoColor), "x\n");
}

TEST(BufferTest, RenderMinimizesEscapes) {
  Buffer buffer(4, Charset::Ascii);
  Style red_bold = Style().Bold().Foreground(Color(255, 0, 0));
  buffer.DrawSymbol(0, 0, 'A', red_bold);
  // Sharing the attributes and changing only the color costs one escape.
  buffer.DrawSymbol(1, 0, 'B', red_bold.Foreground(Color(0, 0, 255)));
  // Dropping bold costs a reset and a fresh start.
  buffer.DrawSymbol(2, 0, 'C', Style().Foreground(Color(0, 0, 255)));
  buffer.DrawSymbol(3, 0, 'D', Style());

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

TEST(BufferTest, RenderEndsRowsWithoutStyle) {
  // A style left on at the end of a row would bleed into whatever the terminal
  // prints next.
  Buffer buffer(4, Charset::Ascii);
  buffer.DrawSymbol(0, 0, 'A', Style().Background(AnsiColor::Red));
  buffer.DrawSymbol(0, 1, 'B', Style().Background(AnsiColor::Red));
  EXPECT_EQ(Render(buffer, ColorMode::Ansi16),
            "\x1b[41mA\x1b[0m\n"
            "\x1b[41mB\x1b[0m\n");
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

  // Measuring starts from where it is told to, which is what the tab stops and
  // the rows a newline returns to are measured from.
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
  // against each other on the text they used to disagree about.
  for (llvm::StringRef text :
       {"hello", "a\tb\tc", "a\nb\r\nc", "中A🔥", "a  b", ""}) {
    Buffer buffer(10, Charset::Utf8);
    EXPECT_EQ(buffer.MeasureText(2, 1, text),
              buffer.DrawText(2, 1, text, Style()))
        << text;
  }
  for (llvm::StringRef text : {"one two three", "a\nlonger line here",
                               "verylongunbreakableword", ""}) {
    Buffer buffer(10, Charset::Utf8);
    EXPECT_EQ(buffer.MeasureWrappedText(2, 1, 8, text),
              buffer.DrawWrappedText(2, 1, 8, text, Style()))
        << text;
  }
}

TEST(BufferTest, WrapWidthIsWhatWrappingDoesNotOverhang) {
  // What `Metrics::WrapWidth` answers is a fact about wrapping, so it is
  // checked here against the wrapping it describes rather than only in
  // `metrics_test`.
  Buffer buffer(10, Charset::Utf8);
  llvm::StringRef text = "some quite long words here";
  int width = buffer.metrics().WrapWidth(text);
  Buffer drawn(1, Charset::Utf8);
  drawn.DrawWrappedText(0, 0, width, text, Style());
  llvm::SmallVector<llvm::StringRef> rows;
  llvm::StringRef(Render(drawn)).split(rows, '\n');
  for (llvm::StringRef row : rows) {
    EXPECT_LE(static_cast<int>(row.size()), width) << row;
  }
}

TEST(BufferTest, BuiltFromCapabilities) {
  Capabilities capabilities;
  capabilities.columns = 5;
  capabilities.charset = Charset::Utf8;

  Buffer buffer(capabilities);
  EXPECT_EQ(buffer.width(), 5);
  EXPECT_EQ(buffer.charset(), Charset::Utf8);
  buffer.DrawHorizontalLine(0, 0, 5, Style());
  EXPECT_EQ(Render(buffer, capabilities.color_mode), "─────\n");

  // With no width to start from, the buffer starts at nothing and grows, which
  // is what it would have done past any starting width anyway.
  capabilities.columns = std::nullopt;
  Buffer grown(capabilities);
  grown.DrawHorizontalLine(0, 0, 5, Style());
  EXPECT_EQ(Render(grown, capabilities.color_mode), "─────\n");
}

TEST(BufferTest, CombiningMarkPastTheWidthItStartedWith) {
  // The buffer grows to hold the text, so a mark arriving past the width it
  // was constructed with attaches to its base like any other.
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawText(0, 0, ("abcde" + AcuteE.drop_front(1)).str(), Style());
  EXPECT_EQ(Render(buffer), ("abcde" + AcuteE.drop_front(1) + "\n").str());
}

TEST(BufferTest, CombiningMarkWithNoBase) {
  // Marks render into the column before them, so one at the start of a row has
  // nowhere to go rather than attaching to the end of the row above.
  Buffer buffer(4, Charset::Utf8);
  buffer.DrawText(0, 0, "ab", Style());
  buffer.DrawText(0, 1, AcuteE.drop_front(1).str(), Style());
  EXPECT_EQ(Render(buffer), "ab\n");
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
  EXPECT_EQ(*read_back, "\x1b[1mhello\x1b[0m\nworld\n");
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

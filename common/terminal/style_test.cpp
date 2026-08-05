// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/style.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Terminal {
namespace {

auto Transition(const Style& from, const Style& to, ColorMode mode)
    -> std::string {
  llvm::SmallString<64> out;
  from.AppendTransitionTo(out, to, mode);
  return std::string(out);
}

// Turning a style on is the transition from the default style, and turning it
// off is the transition back to it.
auto Escapes(const Style& style, ColorMode mode) -> std::string {
  return Transition(Style(), style, mode);
}

auto Reset(const Style& style, ColorMode mode) -> std::string {
  return Transition(style, Style(), mode);
}

TEST(StyleTest, Attributes) {
  EXPECT_EQ(Escapes(Style().Bold(), ColorMode::Truecolor), "\x1b[1m");
  EXPECT_EQ(Escapes(Style().Dim(), ColorMode::Truecolor), "\x1b[2m");
  EXPECT_EQ(Escapes(Style().Italic(), ColorMode::Truecolor), "\x1b[3m");
  EXPECT_EQ(Escapes(Style().Underline(), ColorMode::Truecolor), "\x1b[4m");
  EXPECT_EQ(Escapes(Style().Reverse(), ColorMode::Truecolor), "\x1b[7m");
  EXPECT_EQ(Escapes(Style().Strikethrough(), ColorMode::Truecolor), "\x1b[9m");

  // Attributes combine into a single sequence, in ascending code order.
  EXPECT_EQ(Escapes(Style()
                        .Italic()
                        .Reverse()
                        .Underline(UnderlineShape::Curly)
                        .UnderlineColor(Color(0, 255, 0))
                        .Strikethrough(),
                    ColorMode::Truecolor),
            "\x1b[3;4:3;7;9m\x1b[58;2;0;255;0m");

  // Every attribute at once, which is the most parameters a diff can produce
  // and the bound `AppendDiff` sizes its array for.
  EXPECT_EQ(Escapes(Style()
                        .Bold()
                        .Dim()
                        .Italic()
                        .Underline(UnderlineShape::Double)
                        .Reverse()
                        .Strikethrough(),
                    ColorMode::Truecolor),
            "\x1b[1;2;3;4:2;7;9m");
}

TEST(StyleTest, AttributesAndColors) {
  Style style = Style().Bold().Foreground(Color(255, 0, 0));
  EXPECT_EQ(Escapes(style, ColorMode::Truecolor), "\x1b[1m\x1b[38;2;255;0;0m");
  EXPECT_EQ(Reset(style, ColorMode::Truecolor), "\x1b[0m");

  EXPECT_EQ(Escapes(style, ColorMode::NoColor), "");
  EXPECT_EQ(Reset(style, ColorMode::NoColor), "");

  // A style that set nothing has nothing to reset.
  EXPECT_EQ(Reset(Style(), ColorMode::Truecolor), "");
}

TEST(StyleTest, UnderlineShapesDegradeInAnsi16) {
  // The shaped underlines are colon subparameters, which the terminals limited
  // to 16 colors mishandle. They become plain underlines rather than
  // disappearing, so they still mark what they were meant to mark.
  for (UnderlineShape shape :
       {UnderlineShape::Single, UnderlineShape::Double, UnderlineShape::Curly,
        UnderlineShape::Dotted, UnderlineShape::Dashed}) {
    EXPECT_EQ(Escapes(Style().Underline(shape), ColorMode::Ansi16), "\x1b[4m");
  }

  EXPECT_EQ(
      Escapes(Style().Underline(UnderlineShape::Double), ColorMode::Ansi256),
      "\x1b[4:2m");
  EXPECT_EQ(
      Escapes(Style().Underline(UnderlineShape::Dotted), ColorMode::Truecolor),
      "\x1b[4:4m");
  EXPECT_EQ(
      Escapes(Style().Underline(UnderlineShape::Dashed), ColorMode::Truecolor),
      "\x1b[4:5m");
}

TEST(StyleTest, TransitionAddsWithoutReset) {
  Style red_bold = Style().Bold().Foreground(Color(255, 0, 0));
  Style blue_bold = Style().Bold().Foreground(Color(0, 0, 255));

  // Bold carries over, so only the color has to change.
  EXPECT_EQ(Transition(red_bold, blue_bold, ColorMode::Truecolor),
            "\x1b[38;2;0;0;255m");

  // Adding an attribute needs no reset either.
  EXPECT_EQ(Transition(red_bold, red_bold.Italic(), ColorMode::Truecolor),
            "\x1b[3m");

  // Nor does changing the shape of an underline that is already on.
  EXPECT_EQ(
      Transition(Style().Underline(), Style().Underline(UnderlineShape::Curly),
                 ColorMode::Truecolor),
      "\x1b[4:3m");
}

TEST(StyleTest, TransitionResetsToDrop) {
  Style red_bold = Style().Bold().Foreground(Color(255, 0, 0));

  // Attributes are removed with a reset and a fresh start rather than SGR's
  // entangled off-codes, so dropping bold costs both.
  EXPECT_EQ(Transition(red_bold, Style().Foreground(Color(255, 0, 0)),
                       ColorMode::Truecolor),
            "\x1b[0m\x1b[38;2;255;0;0m");

  // Dropping the color is the same story.
  EXPECT_EQ(Transition(red_bold, Style().Bold(), ColorMode::Truecolor),
            "\x1b[0m\x1b[1m");

  // Returning to no style at all is just the reset.
  EXPECT_EQ(Transition(red_bold, Style(), ColorMode::Truecolor), "\x1b[0m");
}

TEST(StyleTest, TransitionToSelfIsEmpty) {
  Style style = Style().Bold().Italic().Foreground(AnsiColor::Red);
  EXPECT_EQ(Transition(style, style, ColorMode::Truecolor), "");
  EXPECT_EQ(Transition(Style(), Style(), ColorMode::Truecolor), "");
}

TEST(StyleTest, NoColorWritesNothing) {
  Style style = Style().Bold().Foreground(Color(1, 2, 3));
  EXPECT_EQ(Escapes(style, ColorMode::NoColor), "");
  EXPECT_EQ(Reset(style, ColorMode::NoColor), "");
  EXPECT_EQ(Transition(style, Style().Italic(), ColorMode::NoColor), "");
}

TEST(StyleTest, IsVisibleOnBlank) {
  // Attributes that only affect a glyph's own pixels leave a blank cell blank.
  EXPECT_FALSE(Style().IsVisibleOnBlank());
  EXPECT_FALSE(Style().Bold().IsVisibleOnBlank());
  EXPECT_FALSE(Style().Dim().Italic().IsVisibleOnBlank());
  EXPECT_FALSE(Style().Foreground(AnsiColor::Red).IsVisibleOnBlank());

  // These paint the cell itself.
  EXPECT_TRUE(Style().Background(AnsiColor::Red).IsVisibleOnBlank());
  EXPECT_TRUE(Style().Reverse().IsVisibleOnBlank());
  EXPECT_TRUE(Style().Underline().IsVisibleOnBlank());
  EXPECT_TRUE(Style().Strikethrough().IsVisibleOnBlank());
}

TEST(StyleTest, EqualityComparesEveryField) {
  // The comparison is over the bytes of a style rather than its fields, which
  // is only right as long as every byte belongs to a field.
  Style base = Style();
  EXPECT_EQ(base, Style());
  EXPECT_NE(base, base.Bold());
  EXPECT_NE(base, base.Dim());
  EXPECT_NE(base, base.Italic());
  EXPECT_NE(base, base.Reverse());
  EXPECT_NE(base, base.Strikethrough());
  EXPECT_NE(base, base.Underline());
  EXPECT_NE(base, base.Foreground(AnsiColor::Red));
  EXPECT_NE(base, base.Background(AnsiColor::Red));
  EXPECT_NE(base, base.UnderlineColor(AnsiColor::Red));

  // Including fields that differ only in value.
  EXPECT_NE(Style().Foreground(AnsiColor::Red),
            Style().Foreground(AnsiColor::Blue));
  EXPECT_NE(Style().Underline(UnderlineShape::Curly),
            Style().Underline(UnderlineShape::Dotted));

  // And colors that a byte comparison could confuse, as an unset color and
  // these two both hold nothing but zeroes in their channels.
  EXPECT_NE(base, base.Foreground(AnsiColor::Black));
  EXPECT_NE(base, base.Foreground(Color(0, 0, 0)));
  EXPECT_NE(base.Foreground(AnsiColor::Black), base.Foreground(Color(0, 0, 0)));
}

TEST(StyleTest, ColorsCanBeCleared) {
  Style red = Style().Foreground(AnsiColor::Red);
  EXPECT_TRUE(red.foreground().is_set());

  Style cleared = red.Foreground(Color());
  EXPECT_FALSE(cleared.foreground().is_set());
  EXPECT_EQ(cleared, Style());

  // Dropping a color is a reset like dropping an attribute.
  llvm::SmallString<64> bytes;
  red.AppendTransitionTo(bytes, cleared, ColorMode::Ansi16);
  EXPECT_EQ(bytes, "\x1b[0m");
}

TEST(StyleTest, Ansi16Transitions) {
  // Every shape renders as a plain underline here, so switching between two of
  // them is not a change the terminal can see.
  EXPECT_EQ(
      Transition(Style().Underline(), Style().Underline(UnderlineShape::Curly),
                 ColorMode::Ansi16),
      "");

  // The 16 colors still transition normally.
  EXPECT_EQ(Transition(Style().Foreground(AnsiColor::Red),
                       Style().Foreground(AnsiColor::Blue), ColorMode::Ansi16),
            "\x1b[34m");
}

TEST(StyleTest, ChainingLeavesTheOriginalAlone) {
  const Style base = Style().Bold();
  const Style derived = base.Foreground(AnsiColor::Red);

  EXPECT_EQ(base, Style().Bold());
  EXPECT_EQ(derived, Style().Bold().Foreground(AnsiColor::Red));
  EXPECT_NE(base, derived);

  EXPECT_EQ(Style().Bold().Bold(false), Style());
  EXPECT_EQ(Style().Underline().Underline(UnderlineShape::None), Style());
}

TEST(StyleTest, StyledStreaming) {
  RawStringOstream out;
  out << Styled("error", Style().Bold().Foreground(AnsiColor::BrightRed),
                ColorMode::Ansi16)
      << ": bad";
  EXPECT_EQ(out.TakeStr(), "\x1b[1m\x1b[91merror\x1b[0m: bad");

  out << Styled("error", Style().Bold(), ColorMode::NoColor) << ": bad";
  EXPECT_EQ(out.TakeStr(), "error: bad");
}

TEST(StyleTest, Print) {
  EXPECT_EQ(PrintToString(Style()), "Style()");
  EXPECT_EQ(PrintToString(Style().Bold().Foreground(AnsiColor::Red)),
            "Style(bold, foreground=Red)");
  EXPECT_EQ(PrintToString(Style()
                              .Dim()
                              .Italic()
                              .Underline(UnderlineShape::Curly)
                              .UnderlineColor(Color(0, 0, 1))
                              .Background(Color(1, 2, 3))),
            "Style(dim, italic, underline=Curly, background=#010203, "
            "underline_color=#000001)");
}

TEST(StyleTest, EveryPairOfStylesHasATransition) {
  // Every pair of styles has to have a transition, including the ones that
  // drop an attribute and so need a reset rather than a diff.
  std::vector<Style> styles = {
      Style(),
      Style().Bold(),
      Style().Dim().Italic(),
      Style().Reverse().Strikethrough(),
      Style().Underline(UnderlineShape::Curly),
      Style().Underline(UnderlineShape::Double).UnderlineColor(AnsiColor::Red),
      Style().Foreground(AnsiColor::BrightRed).Background(AnsiColor::Black),
      Style().Foreground(Color(1, 2, 3)).Background(Color(250, 251, 252)),
      Style()
          .Bold()
          .Dim()
          .Italic()
          .Reverse()
          .Strikethrough()
          .Underline(UnderlineShape::Dashed)
          .Foreground(Color(9, 9, 9))
          .Background(AnsiColor::White)
          .UnderlineColor(Color(4, 5, 6)),
  };
  for (ColorMode mode : {ColorMode::NoColor, ColorMode::Ansi16,
                         ColorMode::Ansi256, ColorMode::Truecolor}) {
    for (auto [from_index, from] : llvm::enumerate(styles)) {
      for (auto [to_index, to] : llvm::enumerate(styles)) {
        std::string out = Transition(from, to, mode);
        std::string pair =
            llvm::formatv("{0} -> {1}", from_index, to_index).str();
        if (mode == ColorMode::NoColor) {
          // Nothing is said at all when nothing can be shown.
          EXPECT_TRUE(out.empty()) << pair;
        } else if (from == to) {
          // Staying where it already is costs nothing, whatever the mode.
          EXPECT_TRUE(out.empty()) << pair;
        } else if (mode == ColorMode::Truecolor) {
          // Truecolor is the one mode that can express every field, so no two
          // distinct styles render the same and every step has to say
          // something. The narrower modes round colors together, so there two
          // styles can genuinely be one and the transition is empty.
          EXPECT_FALSE(out.empty()) << pair;
        }
      }
    }
  }
}

}  // namespace
}  // namespace Carbon::Terminal

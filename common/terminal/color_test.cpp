// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/color.h"

#include <gtest/gtest.h>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Terminal {
namespace {

auto Escape(Color color, ColorMode mode,
            ColorTarget target = ColorTarget::Foreground) -> std::string {
  llvm::SmallString<32> escape;
  color.AppendEscape(escape, mode, target);
  return std::string(escape);
}

TEST(ColorTest, AnsiEscapes) {
  // Named colors use their own SGR codes rather than the extended forms, in
  // every mode that has color at all, so the terminal renders them from the
  // user's palette.
  for (ColorMode mode :
       {ColorMode::Ansi16, ColorMode::Ansi256, ColorMode::Truecolor}) {
    EXPECT_EQ(Escape(AnsiColor::Red, mode), "\x1b[31m");
    EXPECT_EQ(Escape(AnsiColor::Black, mode), "\x1b[30m");
    EXPECT_EQ(Escape(AnsiColor::BrightCyan, mode), "\x1b[96m");
    EXPECT_EQ(Escape(AnsiColor::Red, mode, ColorTarget::Background),
              "\x1b[41m");
    EXPECT_EQ(Escape(AnsiColor::BrightWhite, mode, ColorTarget::Background),
              "\x1b[107m");
  }

  EXPECT_EQ(Escape(AnsiColor::Red, ColorMode::NoColor), "");
}

TEST(ColorTest, RgbEscapes) {
  Color red(255, 0, 0);
  EXPECT_EQ(Escape(red, ColorMode::Truecolor), "\x1b[38;2;255;0;0m");
  EXPECT_EQ(Escape(red, ColorMode::Truecolor, ColorTarget::Background),
            "\x1b[48;2;255;0;0m");
  EXPECT_EQ(Escape(red, ColorMode::Ansi256), "\x1b[38;5;196m");
  EXPECT_EQ(Escape(red, ColorMode::Ansi16), "\x1b[91m");
  EXPECT_EQ(Escape(red, ColorMode::NoColor), "");
}

TEST(ColorTest, UnderlineEscapes) {
  // Underline colors only exist in the extended-color escapes, so in `Ansi16`
  // the terminal draws the underline in the foreground color.
  EXPECT_EQ(
      Escape(AnsiColor::Red, ColorMode::Truecolor, ColorTarget::Underline),
      "\x1b[58;5;1m");
  EXPECT_EQ(Escape(AnsiColor::Red, ColorMode::Ansi256, ColorTarget::Underline),
            "\x1b[58;5;1m");
  EXPECT_EQ(Escape(AnsiColor::Red, ColorMode::Ansi16, ColorTarget::Underline),
            "");

  Color green(0, 255, 0);
  EXPECT_EQ(Escape(green, ColorMode::Truecolor, ColorTarget::Underline),
            "\x1b[58;2;0;255;0m");
  EXPECT_EQ(Escape(green, ColorMode::Ansi256, ColorTarget::Underline),
            "\x1b[58;5;46m");
  EXPECT_EQ(Escape(green, ColorMode::Ansi16, ColorTarget::Underline), "");
}

TEST(ColorTest, DownsampleToAnsi16) {
  // The reference value of each ANSI color must come back as that color, or
  // downsampling would shift colors that were already expressible. Spelling
  // the values out here rather than reading them back from the same table the
  // implementation uses is what makes this catch a wrong table.
  struct Expected {
    Color color;
    llvm::StringRef escape;
  };
  Expected cases[] = {
      {Color(0, 0, 0), "\x1b[30m"},       {Color(205, 0, 0), "\x1b[31m"},
      {Color(0, 205, 0), "\x1b[32m"},     {Color(205, 205, 0), "\x1b[33m"},
      {Color(0, 0, 238), "\x1b[34m"},     {Color(205, 0, 205), "\x1b[35m"},
      {Color(0, 205, 205), "\x1b[36m"},   {Color(229, 229, 229), "\x1b[37m"},
      {Color(127, 127, 127), "\x1b[90m"}, {Color(255, 0, 0), "\x1b[91m"},
      {Color(0, 255, 0), "\x1b[92m"},     {Color(255, 255, 0), "\x1b[93m"},
      {Color(92, 92, 255), "\x1b[94m"},   {Color(255, 0, 255), "\x1b[95m"},
      {Color(0, 255, 255), "\x1b[96m"},   {Color(255, 255, 255), "\x1b[97m"},
  };
  for (const Expected& expected : cases) {
    EXPECT_EQ(Escape(expected.color, ColorMode::Ansi16), expected.escape)
        << expected.color;
  }

  // Colors between the reference values land on the nearest one.
  EXPECT_EQ(Escape(Color(250, 10, 10), ColorMode::Ansi16), "\x1b[91m");
  EXPECT_EQ(Escape(Color(10, 10, 10), ColorMode::Ansi16), "\x1b[30m");
  EXPECT_EQ(Escape(Color(120, 120, 120), ColorMode::Ansi16), "\x1b[90m");
}

TEST(ColorTest, DownsampleToPalette) {
  // The corners of the 6x6x6 cube are exactly representable.
  EXPECT_EQ(Escape(Color(0, 0, 0), ColorMode::Ansi256), "\x1b[38;5;16m");
  EXPECT_EQ(Escape(Color(255, 255, 255), ColorMode::Ansi256), "\x1b[38;5;231m");
  EXPECT_EQ(Escape(Color(255, 0, 0), ColorMode::Ansi256), "\x1b[38;5;196m");
  EXPECT_EQ(Escape(Color(0, 0, 255), ColorMode::Ansi256), "\x1b[38;5;21m");

  // The cube's levels are unevenly spaced, so rounding has to account for that
  // rather than divide: 95 and 135 are adjacent levels only 40 apart.
  EXPECT_EQ(Escape(Color(95, 0, 0), ColorMode::Ansi256), "\x1b[38;5;52m");
  EXPECT_EQ(Escape(Color(130, 0, 0), ColorMode::Ansi256), "\x1b[38;5;88m");

  // Near-neutral colors land on the gray ramp, which is far finer than the
  // cube's diagonal, except at the ends where the cube wins.
  EXPECT_EQ(Escape(Color(8, 8, 8), ColorMode::Ansi256), "\x1b[38;5;232m");
  EXPECT_EQ(Escape(Color(128, 128, 128), ColorMode::Ansi256), "\x1b[38;5;244m");
  EXPECT_EQ(Escape(Color(238, 238, 238), ColorMode::Ansi256), "\x1b[38;5;255m");
}

TEST(ColorTest, DownsampleAvoidsPaletteEntries) {
  // Indices 0 through 15 render from the user's palette, so an exact RGB
  // request must never be answered with one.
  for (int r = 0; r < 256; r += 17) {
    for (int g = 0; g < 256; g += 17) {
      for (int b = 0; b < 256; b += 17) {
        Color color(r, g, b);
        std::string escape = Escape(color, ColorMode::Ansi256);
        int index = 0;
        ASSERT_TRUE(llvm::to_integer(
            llvm::StringRef(escape).drop_front(7).drop_back(1), index))
            << color;
        EXPECT_GE(index, 16) << color;
      }
    }
  }
}

TEST(ColorTest, Equality) {
  EXPECT_EQ(Color(AnsiColor::Red), Color(AnsiColor::Red));
  EXPECT_NE(Color(AnsiColor::Red), Color(AnsiColor::Blue));
  EXPECT_EQ(Color(1, 2, 3), Color(1, 2, 3));
  EXPECT_NE(Color(1, 2, 3), Color(1, 2, 4));

  // A named color and its reference value are different colors: the terminal
  // renders one from the palette and the other exactly.
  EXPECT_NE(Color(AnsiColor::BrightRed), Color(255, 0, 0));

  // A palette index occupies the same byte as the red channel, so these pairs
  // hold identical channel bytes and are told apart only by their kind.
  EXPECT_NE(Color(AnsiColor::Red), Color(1, 0, 0));
  EXPECT_NE(Color(AnsiColor::Black), Color(0, 0, 0));
  EXPECT_NE(Color(AnsiColor::Black), Color());
  EXPECT_NE(Color(0, 0, 0), Color());
  EXPECT_EQ(Color(), Color());
}

TEST(ColorTest, Unset) {
  EXPECT_FALSE(Color().is_set());
  EXPECT_EQ(Color().kind(), Color::Kind::None);

  // Black is a color like any other, however little of it there is.
  EXPECT_TRUE(Color(AnsiColor::Black).is_set());
  EXPECT_TRUE(Color(0, 0, 0).is_set());
}

TEST(ColorTest, Print) {
  EXPECT_EQ(PrintToString(Color(AnsiColor::BrightMagenta)), "BrightMagenta");
  EXPECT_EQ(PrintToString(Color(0x12, 0xab, 0xff)), "#12abff");
  EXPECT_EQ(PrintToString(Color()), "None");
}

}  // namespace
}  // namespace Carbon::Terminal

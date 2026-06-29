// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/color.h"

#include <algorithm>
#include <array>

#include "common/check.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"

namespace Carbon::Terminal {

// The number of colors in `AnsiColor`.
static constexpr int AnsiColorCount = 16;

// Reference values for the 16 ANSI colors.
//
// Nothing standardizes these: a terminal draws them from the user's palette,
// which is exactly what makes them worth using. But downsampling an RGB color
// still needs some notion of where each named color sits, so these use the
// xterm defaults, which most terminals default to as well.
static constexpr std::array<Color::RgbValue, AnsiColorCount> AnsiColorRgbs = {{
    {.r = 0, .g = 0, .b = 0},        // Black
    {.r = 205, .g = 0, .b = 0},      // Red
    {.r = 0, .g = 205, .b = 0},      // Green
    {.r = 205, .g = 205, .b = 0},    // Yellow
    {.r = 0, .g = 0, .b = 238},      // Blue
    {.r = 205, .g = 0, .b = 205},    // Magenta
    {.r = 0, .g = 205, .b = 205},    // Cyan
    {.r = 229, .g = 229, .b = 229},  // White
    {.r = 127, .g = 127, .b = 127},  // BrightBlack
    {.r = 255, .g = 0, .b = 0},      // BrightRed
    {.r = 0, .g = 255, .b = 0},      // BrightGreen
    {.r = 255, .g = 255, .b = 0},    // BrightYellow
    {.r = 92, .g = 92, .b = 255},    // BrightBlue
    {.r = 255, .g = 0, .b = 255},    // BrightMagenta
    {.r = 0, .g = 255, .b = 255},    // BrightCyan
    {.r = 255, .g = 255, .b = 255},  // BrightWhite
}};

static constexpr std::array<llvm::StringRef, AnsiColorCount> AnsiColorNames = {
    "Black",       "Red",           "Green",       "Yellow",
    "Blue",        "Magenta",       "Cyan",        "White",
    "BrightBlack", "BrightRed",     "BrightGreen", "BrightYellow",
    "BrightBlue",  "BrightMagenta", "BrightCyan",  "BrightWhite"};

// Returns the squared Euclidean distance between two colors, treating the
// channels as orthogonal axes. This is a crude stand-in for perceptual
// distance, but it is predictable and exact matches always win.
static auto DistanceSquared(Color::RgbValue lhs, Color::RgbValue rhs) -> int {
  int dr = static_cast<int>(lhs.r) - static_cast<int>(rhs.r);
  int dg = static_cast<int>(lhs.g) - static_cast<int>(rhs.g);
  int db = static_cast<int>(lhs.b) - static_cast<int>(rhs.b);
  return dr * dr + dg * dg + db * db;
}

// Returns the ANSI color whose reference value is nearest to `rgb`.
static auto NearestAnsiColor(Color::RgbValue rgb) -> AnsiColor {
  int best_index = 0;
  int best_distance = DistanceSquared(rgb, AnsiColorRgbs[0]);
  for (int i = 1; i < AnsiColorCount; ++i) {
    int distance = DistanceSquared(rgb, AnsiColorRgbs[i]);
    if (distance < best_distance) {
      best_distance = distance;
      best_index = i;
    }
  }
  return static_cast<AnsiColor>(best_index);
}

// The channel values of the 6x6x6 color cube at palette indices 16 through
// 231. The first step is much larger than the rest, so a channel can't be
// rounded to the nearest level by dividing.
static constexpr std::array<uint8_t, 6> CubeLevels = {0,   95,  135,
                                                      175, 215, 255};

// The midpoints between adjacent entries of `CubeLevels`, which are where the
// nearest level changes.
static constexpr std::array<uint8_t, 5> CubeLevelMidpoints = {48, 115, 155, 195,
                                                              235};

static_assert(
    [] {
      for (size_t i = 0; i < CubeLevelMidpoints.size(); ++i) {
        // Rounded up, so that a value exactly between two levels takes the
        // higher one.
        if (CubeLevelMidpoints[i] !=
            (CubeLevels[i] + CubeLevels[i + 1] + 1) / 2) {
          return false;
        }
      }
      return true;
    }(),
    "Midpoints must stay in step with the levels they separate.");

// Returns the index into `CubeLevels` of the level nearest `value`.
static auto NearestCubeLevel(uint8_t value) -> int {
  int level = 0;
  while (level < static_cast<int>(CubeLevelMidpoints.size()) &&
         value >= CubeLevelMidpoints[level]) {
    ++level;
  }
  return level;
}

// Returns the 256-color palette index whose color is nearest to `rgb`.
static auto NearestPaletteIndex(Color::RgbValue rgb) -> uint8_t {
  // Only the color cube and the gray ramp are considered. Indices 0 through 15
  // alias the ANSI colors, whose appearance comes from the user's palette, so
  // an exact RGB request must never be answered with one.
  int r_level = NearestCubeLevel(rgb.r);
  int g_level = NearestCubeLevel(rgb.g);
  int b_level = NearestCubeLevel(rgb.b);
  Color::RgbValue cube = {.r = CubeLevels[r_level],
                          .g = CubeLevels[g_level],
                          .b = CubeLevels[b_level]};

  // The gray ramp at indices 232 through 255 runs from 8 to 238 in steps of
  // 10, and is finer than the cube's gray diagonal for near-neutral colors.
  int average = (static_cast<int>(rgb.r) + static_cast<int>(rgb.g) +
                 static_cast<int>(rgb.b)) /
                3;
  int gray_step = std::clamp((average - 8 + 5) / 10, 0, 23);
  auto gray_value = static_cast<uint8_t>(8 + 10 * gray_step);
  Color::RgbValue gray = {.r = gray_value, .g = gray_value, .b = gray_value};

  if (DistanceSquared(rgb, gray) < DistanceSquared(rgb, cube)) {
    return 232 + gray_step;
  }
  return 16 + 36 * r_level + 6 * g_level + b_level;
}

// Returns the SGR parameter selecting `color` for `target`.
//
// The original ANSI codes cover the first eight colors, and the later "bright"
// codes cover the rest at a fixed offset.
static auto AnsiSgrCode(AnsiColor color, ColorTarget target) -> uint8_t {
  CARBON_DCHECK(target != ColorTarget::Underline,
                "Underline color has no direct ANSI form.");
  int index = static_cast<int>(color);
  int base = target == ColorTarget::Background ? 40 : 30;
  if (index >= 8) {
    // Bright foregrounds are 90-97 and bright backgrounds 100-107.
    base += 60;
  }
  return base + (index % 8);
}

// Returns the SGR parameter introducing an extended color for `target`, which
// is followed by either `;5;<index>` or `;2;<r>;<g>;<b>`.
static auto ExtendedSgrCode(ColorTarget target) -> uint8_t {
  switch (target) {
    case ColorTarget::Foreground:
      return 38;
    case ColorTarget::Background:
      return 48;
    case ColorTarget::Underline:
      return 58;
  }
}

auto Color::AppendEscape(OutputBufferRef out, ColorMode mode,
                         ColorTarget target) const -> void {
  if (mode == ColorMode::NoColor) {
    return;
  }

  // Underline colors are only expressible through the extended-color escape,
  // which `Ansi16` doesn't use, so nothing is emitted and the terminal draws
  // the underline in the foreground color.
  if (target == ColorTarget::Underline && mode == ColorMode::Ansi16) {
    return;
  }

  CARBON_DCHECK(is_set(), "Only a color that is set can be selected.");

  if (kind_ == Kind::Ansi) {
    if (target == ColorTarget::Underline) {
      // Named underline colors go through the palette form of the extended
      // escape, as there is no direct code for them.
      out.Append("\x1b[58;5;", static_cast<uint8_t>(ansi()), "m");
    } else {
      out.Append("\x1b[", AnsiSgrCode(ansi(), target), "m");
    }
    return;
  }

  switch (mode) {
    case ColorMode::Truecolor:
      out.Append("\x1b[", ExtendedSgrCode(target), ";2;", channels_.r, ";",
                 channels_.g, ";", channels_.b, "m");
      break;

    case ColorMode::Ansi256:
      out.Append("\x1b[", ExtendedSgrCode(target), ";5;",
                 NearestPaletteIndex(channels_), "m");
      break;

    case ColorMode::Ansi16:
      out.Append("\x1b[", AnsiSgrCode(NearestAnsiColor(channels_), target),
                 "m");
      break;

    case ColorMode::NoColor:
      CARBON_FATAL("Returned above without emitting anything.");
  }
}

auto Color::Print(llvm::raw_ostream& out) const -> void {
  switch (kind_) {
    case Kind::None:
      out << "None";
      return;
    case Kind::Ansi:
      out << AnsiColorNames[static_cast<int>(ansi())];
      return;
    case Kind::Rgb:
      out << llvm::format("#%02x%02x%02x", channels_.r, channels_.g,
                          channels_.b);
      return;
  }
}

}  // namespace Carbon::Terminal

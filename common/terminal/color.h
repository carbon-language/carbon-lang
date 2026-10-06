// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_COLOR_H_
#define CARBON_COMMON_TERMINAL_COLOR_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "common/terminal/output_buffer_ref.h"

namespace Carbon::Terminal {

// The color escape sequences a terminal understands.
//
// Colors, like the other text attributes, are selected with Select Graphic
// Rendition (SGR) escape sequences. The sequence and the codes for the first
// eight colors come from ECMA-48, published in parallel as ANSI X3.64;
// terminal emulators added the bright variants, the 256-color palette, and the
// 24-bit form:
// https://ecma-international.org/publications-and-standards/standards/ecma-48/
//
// These form a ladder: each mode can express everything the modes before it
// can. Colors that the active mode can't express exactly are downsampled to
// the nearest color it can, so callers author in the richest form and let
// rendering degrade on its own.
enum class ColorMode : int8_t {
  // Emit no escape sequences at all, producing plain text.
  NoColor,
  // The 16 colors with SGR codes of their own.
  Ansi16,
  // The 256-color palette: the 16 ANSI colors, a 6x6x6 RGB cube, and a 24-step
  // gray ramp.
  Ansi256,
  // Direct 24-bit RGB, commonly called "truecolor".
  Truecolor,
};

// The 16 colors with SGR codes of their own.
//
// Terminals render these through the user's configured palette, which makes
// them the right choice for output that should blend with the user's theme.
// The tradeoff is that their rendered appearance is outside our control: a
// user's "red" may be any color at all.
enum class AnsiColor : uint8_t {
  Black,
  Red,
  Green,
  Yellow,
  Blue,
  Magenta,
  Cyan,
  White,
  BrightBlack,
  BrightRed,
  BrightGreen,
  BrightYellow,
  BrightBlue,
  BrightMagenta,
  BrightCyan,
  BrightWhite,
};

// Which part of a cell's rendering a color applies to.
enum class ColorTarget : int8_t {
  Foreground,
  Background,
  // The color of the underline itself, independent of the foreground. Only
  // `Ansi256` and richer modes can express this.
  Underline,
};

// A color to render with: one of the 16 named ANSI colors, a 24-bit RGB value,
// or no color at all.
//
// RGB colors render exactly where the terminal supports them, and are
// downsampled where it doesn't. Downsampling to `Ansi16` measures distance
// against fixed reference values, but the terminal renders the result from the
// user's palette, so a downsampled color can land far from the original.
// Prefer `AnsiColor` wherever output should track the user's theme, and RGB
// only where an exact color matters.
//
// A default-constructed color selects nothing. That is how a `Style` spells
// leaving one of its colors to the terminal, so this is a value with an empty
// state rather than something wrapped in an `optional` to get one.
class Color : public Printable<Color> {
 public:
  // Whether a color names a palette entry, gives channel values directly, or
  // selects nothing.
  enum class Kind : uint8_t {
    None,
    Ansi,
    Rgb,
  };

  // The channel values of a 24-bit color.
  struct RgbValue {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    friend auto operator==(RgbValue lhs, RgbValue rhs) -> bool = default;
  };

  constexpr Color() = default;

  // Colors convert implicitly from `AnsiColor` so that call sites can read as
  // `style.Foreground(AnsiColor::Red)`.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr Color(AnsiColor ansi)
      : kind_(Kind::Ansi), channels_{.r = static_cast<uint8_t>(ansi)} {}

  constexpr Color(uint8_t r, uint8_t g, uint8_t b)
      : kind_(Kind::Rgb), channels_{.r = r, .g = g, .b = b} {}

  auto kind() const -> Kind { return kind_; }

  // Returns whether this selects a color at all.
  auto is_set() const -> bool { return kind_ != Kind::None; }

  // Returns the named color. Valid only when `kind()` is `Ansi`.
  auto ansi() const -> AnsiColor {
    CARBON_CHECK(kind_ == Kind::Ansi,
                 "Only a named color has a palette index.");
    return static_cast<AnsiColor>(channels_.r);
  }

  // Returns the channel values. Valid only when `kind()` is `Rgb`.
  auto rgb() const -> RgbValue {
    CARBON_CHECK(kind_ == Kind::Rgb, "Only an RGB color has channel values.");
    return channels_;
  }

  // Appends the escape sequence selecting this color for `target`, which
  // requires that one is set.
  //
  // Appends nothing when `mode` is `NoColor`, or when `target` is `Underline`
  // and `mode` is `Ansi16`, which has no way to express an underline color.
  auto AppendEscape(OutputBufferRef out, ColorMode mode,
                    ColorTarget target) const -> void;

  auto Print(llvm::raw_ostream& out) const -> void;

  // Written out rather than defaulted because the `Printable` base has no
  // comparison of its own, which would leave a defaulted one deleted.
  friend auto operator==(Color lhs, Color rhs) -> bool {
    return lhs.kind_ == rhs.kind_ && lhs.channels_ == rhs.channels_;
  }

 private:
  Kind kind_ = Kind::None;

  // The palette index in `r` with the rest zero for `Ansi`, the channel values
  // for `Rgb`, and all zero for `None`.
  //
  // Overlapping the two in a union would leave the bytes past a palette index
  // unwritten. Every byte carrying part of the value is what lets a whole
  // `Style` be compared as bytes, and an index fits in a channel anyway.
  RgbValue channels_ = {.r = 0, .g = 0, .b = 0};
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_COLOR_H_

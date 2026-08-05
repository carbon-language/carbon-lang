// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_STYLE_H_
#define CARBON_COMMON_TERMINAL_STYLE_H_

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "common/ostream.h"
#include "common/terminal/color.h"
#include "common/terminal/output_buffer_ref.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// The shapes an underline can take.
//
// Only `Single` is universally understood. The rest are selected with
// colon-separated subparameters of the Select Graphic Rendition (SGR)
// underline code, which terminals limited to 16 colors mishandle, so in that
// mode they degrade to `Single` rather than disappearing.
enum class UnderlineShape : int8_t {
  None,
  Single,
  Double,
  Curly,
  Dotted,
  Dashed,
};

// A set of colors and text attributes to render with.
//
// Styles are values, and are composed by chaining, which keeps a named style
// readable at its definition:
//
// ```cpp
// const Style Error = Style().Bold().Foreground(AnsiColor::BrightRed);
// const Style ErrorSquiggle = Error.Underline(UnderlineShape::Curly);
// ```
//
// A style authored for a rich terminal stays meaningful on a poor one. Colors
// the active `ColorMode` can't express are downsampled, an underline shape it
// can't express becomes a plain underline, and an underline color it can't
// express is left to the terminal. `NoColor` drops everything.
class Style : public Printable<Style> {
 public:
  // A default-constructed style sets nothing, and is both the style a terminal
  // starts in and the one it is returned to.
  constexpr Style() = default;

  // Returns this style with the foreground color set, where an unset color
  // leaves the foreground to the terminal.
  auto Foreground(Color color) const -> Style {
    Style result = *this;
    result.foreground_ = color;
    return result;
  }

  // Returns this style with the background color set, where an unset color
  // leaves the background to the terminal.
  auto Background(Color color) const -> Style {
    Style result = *this;
    result.background_ = color;
    return result;
  }

  // Returns this style with the bold attribute set to `value`.
  auto Bold(bool value = true) const -> Style {
    Style result = *this;
    result.bold_ = value;
    return result;
  }

  // Returns this style with the dim (faint) attribute set to `value`.
  auto Dim(bool value = true) const -> Style {
    Style result = *this;
    result.dim_ = value;
    return result;
  }

  // Returns this style with the italic attribute set to `value`.
  auto Italic(bool value = true) const -> Style {
    Style result = *this;
    result.italic_ = value;
    return result;
  }

  // Returns this style with reverse video set to `value`, which has the
  // terminal swap foreground and background when it renders.
  auto Reverse(bool value = true) const -> Style {
    Style result = *this;
    result.reverse_ = value;
    return result;
  }

  // Returns this style with the strikethrough attribute set to `value`.
  auto Strikethrough(bool value = true) const -> Style {
    Style result = *this;
    result.strikethrough_ = value;
    return result;
  }

  // Returns this style with the given underline shape, where
  // `UnderlineShape::None` removes one.
  auto Underline(UnderlineShape shape = UnderlineShape::Single) const -> Style {
    Style result = *this;
    result.underline_shape_ = shape;
    return result;
  }

  // Returns this style with the underline drawn in its own color.
  //
  // Only `Ansi256` and richer modes can express this; elsewhere the underline
  // is drawn in the foreground color, as it is when the color is unset.
  auto UnderlineColor(Color color) const -> Style {
    Style result = *this;
    result.underline_color_ = color;
    return result;
  }

  auto foreground() const -> Color { return foreground_; }
  auto background() const -> Color { return background_; }
  auto underline_color() const -> Color { return underline_color_; }
  auto underline_shape() const -> UnderlineShape { return underline_shape_; }
  auto underline() const -> bool {
    return underline_shape_ != UnderlineShape::None;
  }
  auto bold() const -> bool { return bold_; }
  auto dim() const -> bool { return dim_; }
  auto italic() const -> bool { return italic_; }
  auto reverse() const -> bool { return reverse_; }
  auto strikethrough() const -> bool { return strikethrough_; }

  // Returns whether this style paints anything where there is no glyph.
  //
  // Attributes that only affect a glyph's own pixels, such as the foreground
  // color or weight, are invisible on a blank cell. Rendering uses this to drop
  // trailing blanks, and to decide when a style must be turned off before a
  // newline.
  auto IsVisibleOnBlank() const -> bool {
    return background_.is_set() || reverse_ || strikethrough_ || underline();
  }

  // Appends the escape sequences that move the terminal from this style to
  // `target`.
  //
  // This is the only way styles reach a terminal, because turning a style on
  // and turning it back off are both transitions: from and to the default
  // style respectively.
  //
  // SGR adds attributes one at a time, but its codes for removing them are
  // entangled and unevenly supported: one code clears both bold and dim, and
  // the code some terminals use to clear bold is double-underline in ECMA-48.
  // Dropping anything therefore costs a full reset and a fresh start, so a run
  // is cheapest when every style in it sets the same attributes and the same
  // colors, differing only in the color values.
  auto AppendTransitionTo(OutputBufferRef out, const Style& target,
                          ColorMode mode) const -> void {
    // Rendering calls this for every cell it emits, and with color off no
    // transition can produce anything, so that case is settled here rather than
    // across a call.
    if (mode != ColorMode::NoColor) {
      AppendColorTransitionTo(out, target, mode);
    }
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  // Rendering compares the style of every cell against the one in use, so this
  // compares the bytes rather than member by member. The
  // assertion is what makes that valid: every bit of a style belongs to a
  // member, as the members are all byte-aligned and always initialized, so none
  // of the bytes are padding.
  friend auto operator==(const Style& lhs, const Style& rhs) -> bool {
    static_assert(std::has_unique_object_representations_v<Style>);
    return std::memcmp(&lhs, &rhs, sizeof(Style)) == 0;
  }

 private:
  // Turns every attribute and color off.
  static constexpr llvm::StringRef ResetEscape = "\x1b[0m";

  // Appends the transition to `target`, where `mode` has color.
  auto AppendColorTransitionTo(OutputBufferRef out, const Style& target,
                               ColorMode mode) const -> void;

  // Returns whether `from` sets anything this style leaves unset, which is
  // exactly when the transition needs a reset.
  auto NeedsResetFrom(const Style& from) const -> bool;

  // Appends the escapes taking `from` to this style, which requires that this
  // style drops nothing `from` set: `!NeedsResetFrom(from)`.
  auto AppendDiff(OutputBufferRef out, ColorMode mode, const Style& from) const
      -> void;

  Color foreground_;
  Color background_;
  Color underline_color_;
  UnderlineShape underline_shape_ = UnderlineShape::None;
  bool bold_ = false;
  bool dim_ = false;
  bool italic_ = false;
  bool reverse_ = false;
  bool strikethrough_ = false;
};

// Streams `text` with `style` applied and then removed again:
//
// ```cpp
// out << Styled("error", ErrorStyle, mode) << ": " << message << "\n";
// ```
//
// This is the whole API for output that is a stream of styled runs. Reach for
// `Buffer` instead when output needs to be positioned in two dimensions.
//
// The text is referenced, not copied, so it must outlive the printing.
class Styled : public Printable<Styled> {
 public:
  Styled(llvm::StringRef text, const Style& style, ColorMode mode)
      : text_(text), style_(style), mode_(mode) {}

  auto Print(llvm::raw_ostream& out) const -> void {
    // The escapes and the text they wrap reach the stream as one write, both
    // because that is cheaper and because it keeps a styled run from being
    // split across writes that something else could interleave with.
    llvm::SmallString<128> storage;
    OutputBufferRef bytes = storage;
    Style().AppendTransitionTo(bytes, style_, mode_);
    bytes.Append(text_);
    style_.AppendTransitionTo(bytes, Style(), mode_);
    out << storage;
  }

 private:
  llvm::StringRef text_;
  Style style_;
  ColorMode mode_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_STYLE_H_

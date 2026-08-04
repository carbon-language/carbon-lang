// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_METRICS_H_
#define CARBON_COMMON_TERMINAL_METRICS_H_

#include <array>
#include <cstddef>

#include "common/terminal/capabilities.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// The most bytes one code point encodes to in UTF-8, and storage for one.
inline constexpr size_t MaxUtf8Bytes = 4;
using Utf8Storage = std::array<char, MaxUtf8Bytes>;

// Encodes `symbol` as UTF-8 into `storage`, returning the bytes written.
//
// Code points with no valid encoding, including surrogates and anything past
// U+10FFFF, become the replacement character.
auto EncodeUtf8(char32_t symbol, Utf8Storage& storage) -> llvm::StringRef;

// Returns whether wrapped text can be broken at `c`.
//
// This is the one definition of where a break can fall, so that measuring what
// text wraps into and drawing it wrapped agree about it. Carriage returns
// count, so that a CRLF ending breaks only on its newline.
constexpr auto IsWrapBreak(char c) -> bool {
  return c == ' ' || c == '\t' || c == '\r';
}

// How many columns a terminal spends on text, given the charset it decodes
// with.
//
// Which bytes make up a column depends on the charset, so every question about
// the size of text is a question about the charset as well, and this is what
// answers both at once. `Buffer` holds one and lays its cells out with it;
// anything deciding where to put something asks one directly rather than
// keeping its own idea of how wide a string is.
//
// There is deliberately nothing here that converts between a byte offset and a
// column. Two counts over one string that don't convert implicitly are the one
// pair in rendering that must not be taken for each other -- which byte a
// column lands on depends on the encoding, and which column a byte lands in
// depends on the width of everything before it -- and the mistake is invisible
// when it happens, moving a caret a few columns on the inputs nobody tests.
// `TakeColumns` is what a caller wanting to cut text at a column uses, and it
// hands back the text rather than an offset into it, so the two counts never
// meet.
class Metrics {
 public:
  explicit constexpr Metrics(Charset charset) : charset_(charset) {}

  constexpr auto charset() const -> Charset { return charset_; }

  // Removes the next symbol from `text`, which must not be empty, and returns
  // it: one byte under `Charset::Ascii`, and one decoded code point under
  // `Charset::Utf8`.
  //
  // A byte that doesn't start a valid sequence yields the replacement
  // character and is consumed on its own, so decoding resynchronizes at the
  // next byte rather than discarding the rest of the text.
  auto TakeSymbol(llvm::StringRef& text) const -> char32_t;

  // Returns the columns `symbol` occupies once drawn, which is what drawing it
  // advances by. A combining mark is zero, since it renders into the column
  // before it, and anything with no printable rendering is one, since it is
  // drawn as a replacement character.
  auto SymbolWidth(char32_t symbol) const -> int;

  // Returns the symbol a terminal decoding this charset is actually given for
  // `symbol`, which is a replacement character where it has no dependable
  // rendering of its own.
  //
  // Under `Charset::Ascii` that is anything outside printable ASCII, because a
  // terminal decoding some single-byte encoding will draw such a byte as
  // something and there is no way to know what. Under `Charset::Utf8` it is
  // anything with no printable rendering at all, including invalid UTF-8.
  auto RenderedSymbol(char32_t symbol) const -> char32_t;

  // Returns the columns `text` occupies once drawn.
  //
  // `text` must hold no character that drawing gives a width other than its
  // symbols', so no tab, newline, or carriage return. Those are positional --
  // what a tab advances by depends on where the text began -- which makes them
  // questions about a drawing rather than about the text, and `Buffer` answers
  // those.
  auto Width(llvm::StringRef text) const -> int;

  // Returns the fewest columns `text` wraps into without overhanging them,
  // which is the width of its widest word since wrapping never breaks one.
  //
  // Wrapping into fewer still draws everything, so this is what to ask when a
  // layout has somewhere else to put text that doesn't fit, rather than a
  // reason to drop it.
  auto WrapWidth(llvm::StringRef text) const -> int;

  // Removes and returns the longest prefix of `text` that occupies at most
  // `columns` columns.
  //
  // A symbol that would straddle the end stops the walk before it, so a cut
  // never lands inside one and the prefix is never wider than asked for -- it
  // can be one column narrower, where a double-width symbol sits on the
  // boundary. `Width` on the result is what says which it was.
  auto TakeColumns(llvm::StringRef& text, int columns) const -> llvm::StringRef;

 private:
  Charset charset_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_METRICS_H_

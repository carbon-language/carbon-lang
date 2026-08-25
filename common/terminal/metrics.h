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

// Encodes `code_point` as UTF-8 into `storage`, returning the bytes written.
//
// Code points with no valid encoding, including surrogates and anything past
// U+10FFFF, become the replacement character.
auto EncodeUtf8(char32_t code_point, Utf8Storage& storage) -> llvm::StringRef;

// How many columns a terminal spends on text, given the charset it decodes
// with.
//
// Which bytes make up a column depends on the charset, so every question about
// the size of text is a question about the charset as well, and this is what
// answers both at once. `Buffer` holds one and lays its cells out with it;
// anything deciding where to put something asks one directly rather than
// keeping its own idea of how wide a string is.
//
// Nothing here converts between a byte offset and a column: which byte a column
// lands on depends on the encoding, and which column a byte lands in depends on
// the width of everything before it. `TakeColumns` hands back the text it cut
// rather than an offset into it, so a caller never holds one count where the
// other belongs.
//
// TODO: Every width here is a sum over code points taken in logical order,
// which is only the width on screen for left-to-right text. Bidirectional text
// reorders, so a run's width still adds up but `TakeColumns` has no meaning:
// the prefix occupying the first N columns need not be a prefix of the string.
// Settle this together with the question `Buffer`'s own TODO describes, since
// both turn on what a client hands over.
class Metrics {
 public:
  explicit constexpr Metrics(Charset charset) : charset_(charset) {}

  constexpr auto charset() const -> Charset { return charset_; }

  // Removes the next code point from `text`, which must not be empty, and
  // returns it: one byte under `Charset::Ascii`, and one decoded code point
  // under `Charset::Utf8`.
  //
  // A byte that doesn't start a valid sequence yields the replacement
  // character and is consumed on its own, so decoding resynchronizes at the
  // next byte rather than discarding the rest of the text.
  auto TakeCodePoint(llvm::StringRef& text) const -> char32_t;

  // Returns the columns `code_point` occupies once drawn, which is what drawing
  // it advances by.
  //
  // Under `Charset::Ascii` every code point is one column. Under
  // `Charset::Utf8` a combining mark is zero, since it renders into the column
  // before it, and anything with no printable rendering is one, since it is
  // drawn as a replacement character.
  //
  // A combining mark is the only thing zero is ever the answer for, which is
  // what lets `Buffer` read a zero as one: a code point to fold into the cell
  // before it rather than give a cell of its own. A code point that takes no
  // column without combining with anything, such as U+200C ZERO WIDTH
  // NON-JOINER, has no printable rendering here and takes the column its
  // replacement character does. Terminals disagree about those -- some give
  // them a column and some don't -- so drawing one as itself would leave the
  // columns counted here and the columns painted disagreeing from there on.
  auto CodePointWidth(char32_t code_point) const -> int;

  // Returns the code point to render for `code_point`, which is a replacement
  // character where it has no dependable rendering of its own.
  //
  // Under `Charset::Ascii` that is anything outside printable ASCII, because a
  // terminal decoding some single-byte encoding will draw such a byte as
  // something and there is no way to know what. Under `Charset::Utf8` it is
  // anything with no printable rendering at all, which includes the surrogates
  // and so covers everything UTF-8 has no encoding for as well.
  auto RenderedCodePoint(char32_t code_point) const -> char32_t;

  // Returns the columns `text` occupies once drawn.
  //
  // `text` must hold no character that drawing gives a width other than its
  // code points', so no tab, newline, or carriage return. Those are positional
  // -- what a tab advances by depends on where the text began -- which makes
  // them questions about a drawing rather than about the text, and `Buffer`
  // answers those.
  auto Width(llvm::StringRef text) const -> int;

  // Removes and returns the longest prefix of `text` that occupies at most
  // `columns` columns.
  //
  // A code point that would straddle the end stops the walk before it, so a cut
  // never lands inside one and the prefix is never wider than asked for -- it
  // can be one column narrower, where a double-width character sits on the
  // boundary.
  auto TakeColumns(llvm::StringRef& text, int columns) const -> llvm::StringRef;

 private:
  Charset charset_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_METRICS_H_

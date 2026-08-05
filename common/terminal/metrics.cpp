// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/metrics.h"

#include <algorithm>

#include "common/check.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Unicode.h"

namespace Carbon::Terminal {

// Stands in for anything a UTF-8 terminal has no rendering for: invalid UTF-8,
// control characters, and unassigned code points.
static constexpr char32_t Utf8Replacement = U'�';

// Returns whether an ASCII terminal renders `code_point` as itself, in one
// column.
static auto IsPrintableAscii(char32_t code_point) -> bool {
  return code_point >= 0x20 && code_point < 0x7f;
}

auto EncodeUtf8(char32_t code_point, Utf8Storage& storage) -> llvm::StringRef {
  // Most of what gets rendered is ASCII, and encoding it is a single byte.
  if (code_point < 0x80) {
    storage[0] = static_cast<char>(code_point);
    return llvm::StringRef(storage.data(), 1);
  }

  // Surrogates have no encoding of their own, and nothing past the last code
  // point has one at all.
  if (code_point > 0x10ffff || (code_point >= 0xd800 && code_point < 0xe000)) {
    code_point = Utf8Replacement;
  }

  // Spelled out rather than handed to a general converter, which walks a range
  // and checks bounds this already knows. Box-drawing characters go through
  // here for every cell of every line drawn.
  auto trailing = [code_point](int shift) {
    return static_cast<char>(0x80 | ((code_point >> shift) & 0x3f));
  };
  if (code_point < 0x800) {
    storage[0] = static_cast<char>(0xc0 | (code_point >> 6));
    storage[1] = trailing(0);
    return llvm::StringRef(storage.data(), 2);
  }
  if (code_point < 0x10000) {
    storage[0] = static_cast<char>(0xe0 | (code_point >> 12));
    storage[1] = trailing(6);
    storage[2] = trailing(0);
    return llvm::StringRef(storage.data(), 3);
  }
  storage[0] = static_cast<char>(0xf0 | (code_point >> 18));
  storage[1] = trailing(12);
  storage[2] = trailing(6);
  storage[3] = trailing(0);
  return llvm::StringRef(storage.data(), 4);
}

// Returns the columns `code_point` occupies on a UTF-8 terminal: zero for a
// combining mark, one or two for one with a glyph of its own, and a
// negative value when there is no printable rendering for it.
//
// TODO: This encodes a code point only for LLVM to decode it again.
// `llvm::sys::unicode::charWidth` computes exactly this and is what
// `columnWidthUTF8` calls once per code point, but it is file-local to LLVM's
// `Unicode.cpp`. Exposing it there would let this call it directly. LLVM's own
// contract already says a string's width is the sum of its code points', so
// there is nothing in the way of it.
static auto Utf8CodePointWidth(char32_t code_point) -> int {
  // Printable ASCII is one column, and is most of what gets measured. The
  // general path parses a UTF-8 sequence and searches several code point
  // range tables, which is far more than this needs.
  if (IsPrintableAscii(code_point)) {
    return 1;
  }

  Utf8Storage storage;
  return llvm::sys::unicode::columnWidthUTF8(EncodeUtf8(code_point, storage));
}

// Removes the first UTF-8 sequence from `text` and returns the code point it
// encodes.
static auto TakeUtf8CodePoint(llvm::StringRef& text) -> char32_t {
  const auto* begin = reinterpret_cast<const llvm::UTF8*>(text.data());
  const auto* pos = begin;
  llvm::UTF32 code_point = 0;
  if (llvm::convertUTF8Sequence(&pos, begin + text.size(), &code_point,
                                llvm::strictConversion) != llvm::conversionOK) {
    text = text.drop_front(1);
    return Utf8Replacement;
  }
  text = text.drop_front(pos - begin);
  return code_point;
}

auto Metrics::TakeCodePoint(llvm::StringRef& text) const -> char32_t {
  CARBON_CHECK(!text.empty(), "No code point to take.");
  if (charset_ == Charset::Ascii) {
    auto byte = static_cast<unsigned char>(text.front());
    text = text.drop_front();
    return byte;
  }
  return TakeUtf8CodePoint(text);
}

auto Metrics::CodePointWidth(char32_t code_point) const -> int {
  if (charset_ == Charset::Ascii) {
    return 1;
  }
  int width = Utf8CodePointWidth(code_point);
  // A code point with no rendering is drawn as the replacement character, which
  // takes one column.
  return width < 0 ? 1 : width;
}

auto Metrics::RenderedCodePoint(char32_t code_point) const -> char32_t {
  if (charset_ == Charset::Ascii) {
    return IsPrintableAscii(code_point) ? code_point : U'?';
  }
  return Utf8CodePointWidth(code_point) < 0 ? Utf8Replacement : code_point;
}

auto Metrics::Width(llvm::StringRef text) const -> int {
  // Checked rather than debug-checked: text with one of these in it measures as
  // though each took one column, which is not what drawing does, and measuring
  // wrong is invisible in the output. The scan is one more linear pass over
  // text that is walked linearly anyway.
  CARBON_CHECK(
      text.find_first_of("\t\n\r") == llvm::StringRef::npos,
      "Width is only for text whose width is its code points', but got `{0}`.",
      text);
  if (charset_ == Charset::Ascii) {
    return static_cast<int>(text.size());
  }

  // Text that is valid UTF-8 throughout and printable throughout is the common
  // case, and LLVM measures a whole run of it in one pass. It answers with a
  // negative value rather than a width when the text holds anything it can't
  // measure, which is what the walk below is for: each such code point still
  // takes the one column the replacement character drawn for it will.
  int width = llvm::sys::unicode::columnWidthUTF8(text);
  if (width >= 0) {
    return width;
  }

  width = 0;
  while (!text.empty()) {
    width += CodePointWidth(TakeUtf8CodePoint(text));
  }
  return width;
}

auto Metrics::WrapWidth(llvm::StringRef text) const -> int {
  int width = 0;
  while (!text.empty()) {
    llvm::StringRef word =
        text.take_until([](char c) { return c == '\n' || IsWrapBreak(c); });
    width = std::max(width, Width(word));
    text = text.drop_front(std::max<size_t>(word.size(), 1));
  }
  return width;
}

auto Metrics::TakeColumns(llvm::StringRef& text, int columns) const
    -> llvm::StringRef {
  llvm::StringRef rest = text;
  int taken = 0;
  while (!rest.empty()) {
    llvm::StringRef next = rest;
    int width = CodePointWidth(TakeCodePoint(next));
    if (taken + width > columns) {
      break;
    }
    taken += width;
    rest = next;
  }
  llvm::StringRef prefix = text.drop_back(rest.size());
  text = rest;
  return prefix;
}

}  // namespace Carbon::Terminal

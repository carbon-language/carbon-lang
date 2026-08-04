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

// Returns whether an ASCII terminal renders `symbol` as itself, in one column.
static auto IsPrintableAscii(char32_t symbol) -> bool {
  return symbol >= 0x20 && symbol < 0x7f;
}

auto EncodeUtf8(char32_t symbol, Utf8Storage& storage) -> llvm::StringRef {
  // Most of what gets rendered is ASCII, and encoding it is a single byte.
  if (symbol < 0x80) {
    storage[0] = static_cast<char>(symbol);
    return llvm::StringRef(storage.data(), 1);
  }

  // Surrogates have no encoding of their own, and nothing past the last code
  // point has one at all.
  if (symbol > 0x10ffff || (symbol >= 0xd800 && symbol < 0xe000)) {
    symbol = Utf8Replacement;
  }

  // Spelled out rather than handed to a general converter, which walks a range
  // and checks bounds this already knows. Box-drawing characters go through
  // here for every cell of every line drawn.
  auto trailing = [symbol](int shift) {
    return static_cast<char>(0x80 | ((symbol >> shift) & 0x3f));
  };
  if (symbol < 0x800) {
    storage[0] = static_cast<char>(0xc0 | (symbol >> 6));
    storage[1] = trailing(0);
    return llvm::StringRef(storage.data(), 2);
  }
  if (symbol < 0x10000) {
    storage[0] = static_cast<char>(0xe0 | (symbol >> 12));
    storage[1] = trailing(6);
    storage[2] = trailing(0);
    return llvm::StringRef(storage.data(), 3);
  }
  storage[0] = static_cast<char>(0xf0 | (symbol >> 18));
  storage[1] = trailing(12);
  storage[2] = trailing(6);
  storage[3] = trailing(0);
  return llvm::StringRef(storage.data(), 4);
}

// Returns the columns `symbol` occupies on a UTF-8 terminal: zero for a
// combining mark, one or two for a symbol with a glyph of its own, and a
// negative value when there is no printable rendering for it.
static auto Utf8SymbolWidth(char32_t symbol) -> int {
  // Printable ASCII is one column, and is most of what gets measured. The
  // general path parses a UTF-8 sequence and searches several code point
  // range tables, which is far more than this needs.
  if (IsPrintableAscii(symbol)) {
    return 1;
  }

  Utf8Storage storage;
  return llvm::sys::unicode::columnWidthUTF8(EncodeUtf8(symbol, storage));
}

// Removes the first UTF-8 sequence from `text` and returns the code point it
// encodes.
static auto TakeUtf8Symbol(llvm::StringRef& text) -> char32_t {
  const auto* begin = reinterpret_cast<const llvm::UTF8*>(text.data());
  const auto* pos = begin;
  llvm::UTF32 symbol = 0;
  if (llvm::convertUTF8Sequence(&pos, begin + text.size(), &symbol,
                                llvm::strictConversion) != llvm::conversionOK) {
    text = text.drop_front(1);
    return Utf8Replacement;
  }
  text = text.drop_front(pos - begin);
  return symbol;
}

auto Metrics::TakeSymbol(llvm::StringRef& text) const -> char32_t {
  CARBON_DCHECK(!text.empty());
  if (charset_ == Charset::Ascii) {
    auto byte = static_cast<unsigned char>(text.front());
    text = text.drop_front();
    return byte;
  }
  return TakeUtf8Symbol(text);
}

auto Metrics::SymbolWidth(char32_t symbol) const -> int {
  if (charset_ == Charset::Ascii) {
    return 1;
  }
  int width = Utf8SymbolWidth(symbol);
  // A symbol with no rendering is drawn as the replacement character, which
  // takes one column.
  return width < 0 ? 1 : width;
}

auto Metrics::RenderedSymbol(char32_t symbol) const -> char32_t {
  if (charset_ == Charset::Ascii) {
    return IsPrintableAscii(symbol) ? symbol : U'?';
  }
  return Utf8SymbolWidth(symbol) < 0 ? Utf8Replacement : symbol;
}

auto Metrics::Width(llvm::StringRef text) const -> int {
  CARBON_DCHECK(
      !text.contains('\t') && !text.contains('\n') && !text.contains('\r'),
      "Width is only for text whose width is its symbols', but got {0:?}.",
      text);
  if (charset_ == Charset::Ascii) {
    return static_cast<int>(text.size());
  }

  int width = 0;
  while (!text.empty()) {
    width += SymbolWidth(TakeUtf8Symbol(text));
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
    int width = SymbolWidth(TakeSymbol(next));
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

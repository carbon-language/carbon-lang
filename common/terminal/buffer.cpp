// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <array>

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Unicode.h"

namespace Carbon::Terminal {

// Stands in for a byte an ASCII terminal has no dependable rendering for.
static constexpr char32_t AsciiReplacement = U'?';

// Stands in for anything a UTF-8 terminal has no rendering for: invalid UTF-8,
// control characters, and unassigned code points.
static constexpr char32_t Utf8Replacement = U'�';

// Columns between tab stops, measured from where the text began rather than
// from the left edge of the buffer.
static constexpr int TabWidth = 8;

// The most bytes of combining marks kept on one cell. Text stacking more than
// this is either adversarial or already illegible, and keeping all of it would
// let one input character produce unbounded output.
static constexpr size_t MaxCombiningBytes = 32;

// The most bytes one code point encodes to in UTF-8.
static constexpr size_t MaxUtf8Bytes = 4;

// Returns whether an ASCII terminal renders `symbol` as itself, in one column.
static auto IsPrintableAscii(char32_t symbol) -> bool {
  return symbol >= 0x20 && symbol < 0x7f;
}

// Encodes `symbol` as UTF-8 into `storage`, returning the bytes written.
//
// Code points with no valid encoding, including surrogates and anything past
// U+10FFFF, become the replacement character.
static auto EncodeUtf8(char32_t symbol, std::array<char, MaxUtf8Bytes>& storage)
    -> llvm::StringRef {
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

  std::array<char, MaxUtf8Bytes> storage;
  return llvm::sys::unicode::columnWidthUTF8(EncodeUtf8(symbol, storage));
}

// Removes the first UTF-8 sequence from `text` and returns the code point it
// encodes.
//
// A byte that doesn't start a valid sequence yields the replacement character
// and is consumed on its own, so decoding resynchronizes at the next byte
// rather than discarding the rest of the text.
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

// Glyphs for every combination of line directions, indexed by the direction
// bits. Index zero never comes up, as a cell with no directions holds no line.
static constexpr std::array<char32_t, 16> Utf8LineGlyphs = {
    U' ',  // (none)
    U'─',  // left
    U'─',  // right
    U'─',  // left, right
    U'│',  // up
    U'╯',  // left, up
    U'╰',  // right, up
    U'┴',  // left, right, up
    U'│',  // down
    U'╮',  // left, down
    U'╭',  // right, down
    U'┬',  // left, right, down
    U'│',  // up, down
    U'┤',  // left, up, down
    U'├',  // right, up, down
    U'┼',  // left, right, up, down
};

// The ASCII stand-ins, which can only distinguish horizontal, vertical, and
// everything else.
static constexpr std::array<char32_t, 16> AsciiLineGlyphs = {
    U' ', U'-', U'-', U'-', U'|', U'+', U'+', U'+',
    U'|', U'+', U'+', U'+', U'|', U'+', U'+', U'+',
};

Buffer::Buffer(int width, Charset charset) : width_(width), charset_(charset) {
  CARBON_CHECK(width > 0, "Buffer width must be positive, but was {0}.", width);
}

auto Buffer::TakeSymbol(llvm::StringRef& text) const -> char32_t {
  CARBON_DCHECK(!text.empty());
  if (charset_ == Charset::Ascii) {
    auto byte = static_cast<unsigned char>(text.front());
    text = text.drop_front();
    return byte;
  }
  return TakeUtf8Symbol(text);
}

auto Buffer::SymbolWidth(char32_t symbol) const -> int {
  if (charset_ == Charset::Ascii) {
    return 1;
  }
  int width = Utf8SymbolWidth(symbol);
  // A symbol with no rendering is drawn as the replacement character, which
  // takes one column.
  return width < 0 ? 1 : width;
}

auto Buffer::MeasureWidth(llvm::StringRef text) const -> int {
  if (charset_ == Charset::Ascii) {
    return static_cast<int>(text.size());
  }

  int width = 0;
  while (!text.empty()) {
    width += SymbolWidth(TakeUtf8Symbol(text));
  }
  return width;
}

auto Buffer::height() const -> int {
  return static_cast<int>(cells_.size()) / width_;
}

auto Buffer::EnsureRow(int y) -> void {
  if (y < height()) {
    return;
  }
  cells_.resize(static_cast<size_t>(y + 1) * width_);
}

auto Buffer::ClearCells(int x, int y, int width) -> void {
  // A cleared range must not leave half of a double-width symbol behind, so it
  // extends over either half that crosses its edges.
  int begin = x;
  if (begin > 0 && CellAt(begin, y).is_continuation) {
    --begin;
  }
  int end = x + width;
  if (end < width_ && CellAt(end, y).is_continuation) {
    ++end;
  }

  for (int i = begin; i < end; ++i) {
    CellAt(i, y) = Cell();
    combining_marks_.erase(CellIndex(i, y));
  }
}

auto Buffer::AttachCombiningMark(int x, int y, char32_t symbol) -> void {
  // Marks render into the column before them, so there must be one, it must be
  // in the buffer, and it must already have been drawn. Text walks past the
  // right edge rather than stopping there, so `x` can be well beyond it, in
  // which case the base was clipped and its marks go with it.
  if (x <= 0 || x > width_ || y < 0 || y >= height()) {
    return;
  }
  int base = x - 1;
  if (CellAt(base, y).is_continuation) {
    --base;
  }

  std::array<char, MaxUtf8Bytes> storage;
  llvm::StringRef encoded = EncodeUtf8(symbol, storage);
  std::string& marks = combining_marks_[CellIndex(base, y)];
  if (marks.size() + encoded.size() > MaxCombiningBytes) {
    return;
  }
  marks.append(encoded.data(), encoded.size());
}

auto Buffer::DrawSymbol(int x, int y, char32_t symbol, const Style& style)
    -> int {
  if (charset_ == Charset::Ascii) {
    if (x >= 0 && x < width_ && y >= 0) {
      EnsureRow(y);
      CellAt(x, y) = {
          .symbol = IsPrintableAscii(symbol) ? symbol : AsciiReplacement,
          .style = style};
    }
    return 1;
  }

  int width = Utf8SymbolWidth(symbol);
  if (width == 0) {
    AttachCombiningMark(x, y, symbol);
    return 0;
  }
  if (width < 0) {
    symbol = Utf8Replacement;
    width = 1;
  }

  // Out of bounds, or a double-width symbol that would straddle the right
  // edge; splitting one would leave the terminal rendering half a character.
  if (x < 0 || y < 0 || x + width > width_) {
    return width;
  }

  EnsureRow(y);
  ClearCells(x, y, width);

  Cell& cell = CellAt(x, y);
  cell.symbol = symbol;
  cell.style = style;
  for (int i = 1; i < width; ++i) {
    Cell& continuation = CellAt(x + i, y);
    continuation.style = style;
    continuation.is_continuation = true;
  }
  return width;
}

auto Buffer::DrawLine(int x, int y, uint8_t directions, const Style& style)
    -> void {
  if (x < 0 || x >= width_ || y < 0) {
    return;
  }
  CARBON_DCHECK(directions < Utf8LineGlyphs.size());
  EnsureRow(y);

  uint8_t existing = CellAt(x, y).lines;
  if (existing == 0) {
    // Whatever is here isn't a line. Clearing also removes either half of a
    // double-width symbol the cell was part of.
    ClearCells(x, y, 1);
  }

  Cell& cell = CellAt(x, y);
  cell.lines = existing | directions;
  cell.symbol = charset_ == Charset::Utf8 ? Utf8LineGlyphs[cell.lines]
                                          : AsciiLineGlyphs[cell.lines];
  cell.style = style;
}

auto Buffer::DrawHorizontalLine(int x, int y, int length, const Style& style)
    -> void {
  for (int i = 0; i < length; ++i) {
    // The ends of a line only connect inward, so a line meeting another at its
    // end forms a corner rather than a crossing. A one-cell line has no inward
    // direction and is drawn as a bare horizontal segment.
    uint8_t directions =
        length == 1 ? LineLeft | LineRight
                    : (i > 0 ? LineLeft : 0) | (i + 1 < length ? LineRight : 0);
    DrawLine(x + i, y, directions, style);
  }
}

auto Buffer::DrawVerticalLine(int x, int y, int length, const Style& style)
    -> void {
  for (int i = 0; i < length; ++i) {
    uint8_t directions =
        length == 1 ? LineUp | LineDown
                    : (i > 0 ? LineUp : 0) | (i + 1 < length ? LineDown : 0);
    DrawLine(x, y + i, directions, style);
  }
}

auto Buffer::DrawBox(int x, int y, int box_width, int box_height,
                     const Style& style) -> void {
  if (box_width <= 0 || box_height <= 0) {
    return;
  }
  // A box with no interior is just the one line that bounds it.
  if (box_width == 1) {
    DrawVerticalLine(x, y, box_height, style);
    return;
  }
  if (box_height == 1) {
    DrawHorizontalLine(x, y, box_width, style);
    return;
  }

  DrawHorizontalLine(x, y, box_width, style);
  DrawHorizontalLine(x, y + box_height - 1, box_width, style);
  DrawVerticalLine(x, y, box_height, style);
  DrawVerticalLine(x + box_width - 1, y, box_height, style);
}

auto Buffer::DrawText(int x, int y, llvm::StringRef text, const Style& style)
    -> int {
  int cur_x = x;
  int cur_y = y;

  while (!text.empty()) {
    char32_t symbol = TakeSymbol(text);
    if (symbol == '\n') {
      cur_x = x;
      ++cur_y;
      continue;
    }
    if (symbol == '\r') {
      cur_x = x;
      continue;
    }
    if (symbol == '\t') {
      int stop = x + ((cur_x - x) / TabWidth + 1) * TabWidth;
      for (; cur_x < stop; ++cur_x) {
        DrawSymbol(cur_x, cur_y, ' ', style);
      }
      continue;
    }

    cur_x += DrawSymbol(cur_x, cur_y, symbol, style);
  }

  return cur_y - y + 1;
}

// Returns whether `c` is a character wrapped text can be broken at. Carriage
// returns count, so that a CRLF ending breaks only on its newline.
static auto IsBreakSpace(char c) -> bool {
  return c == ' ' || c == '\t' || c == '\r';
}

auto Buffer::DrawWrappedText(int x, int y, int max_width, llvm::StringRef text,
                             const Style& style) -> int {
  if (max_width <= 0) {
    return 0;
  }

  int cur_x = x;
  int cur_y = y;
  int limit = x + max_width;

  // Splitting on bytes is safe because every character text can break at is
  // ASCII, and UTF-8 never encodes anything else using an ASCII byte. Only the
  // runs that get drawn are decoded.
  while (!text.empty()) {
    if (text.front() == '\n') {
      text = text.drop_front();
      cur_x = x;
      ++cur_y;
      continue;
    }

    if (IsBreakSpace(text.front())) {
      llvm::StringRef spaces = text.take_while(IsBreakSpace);
      text = text.drop_front(spaces.size());
      // Spaces are dropped wherever they would begin a row, which is what
      // keeps wrapped text aligned with where it started. Carriage returns are
      // never drawn, so CRLF line endings break exactly once.
      if (cur_x > x) {
        for (char space : spaces) {
          if (space == '\r' || cur_x >= limit) {
            continue;
          }
          cur_x += DrawSymbol(cur_x, cur_y, ' ', style);
        }
      }
      continue;
    }

    llvm::StringRef word =
        text.take_until([](char c) { return c == '\n' || IsBreakSpace(c); });
    text = text.drop_front(word.size());

    // Move a word that doesn't fit down to the next row. If it doesn't fit
    // there either, the loop below breaks it as it draws it.
    if (cur_x > x && cur_x + MeasureWidth(word) > limit) {
      cur_x = x;
      ++cur_y;
    }

    while (!word.empty()) {
      char32_t symbol = TakeSymbol(word);
      // Combining marks have no width and must stay with the symbol they
      // follow, so they never trigger a break.
      int width = SymbolWidth(symbol);
      // A symbol too wide for the whole wrap region has no row that could hold
      // it, and drawing it anyway would run past the region into whatever is
      // laid out beside it.
      if (width > max_width) {
        continue;
      }
      if (width > 0 && cur_x > x && cur_x + width > limit) {
        cur_x = x;
        ++cur_y;
      }
      cur_x += DrawSymbol(cur_x, cur_y, symbol, style);
    }
  }

  return cur_y - y + 1;
}

auto Buffer::LastVisibleColumn(int y, ColorMode mode) const -> int {
  // A style only paints a blank cell if it is rendered at all, so with color
  // off a blank cell is padding whatever style it carries.
  bool styles_render = mode != ColorMode::NoColor;
  for (int x = width_ - 1; x >= 0; --x) {
    const Cell& cell = CellAt(x, y);
    if (cell.is_continuation || cell.symbol != ' ' ||
        (styles_render && cell.style.IsVisibleOnBlank()) ||
        (!combining_marks_.empty() &&
         combining_marks_.contains(CellIndex(x, y)))) {
      return x;
    }
  }
  return -1;
}

auto Buffer::Render(OutputBufferRef out, ColorMode mode) const -> void {
  std::array<char, MaxUtf8Bytes> storage;

  // The style a terminal starts a row in, and the one each row leaves it in.
  const Style default_style;

  for (int y = 0, rows = height(); y < rows; ++y) {
    // Cells outlive this loop, so the active style is tracked by pointing at
    // one rather than copying a whole style per cell.
    const Style* active = &default_style;
    int last = LastVisibleColumn(y, mode);
    for (int x = 0; x <= last; ++x) {
      const Cell& cell = CellAt(x, y);
      if (cell.is_continuation) {
        continue;
      }

      active->AppendTransitionTo(out, cell.style, mode);
      active = &cell.style;
      out.Append(EncodeUtf8(cell.symbol, storage));

      // Almost nothing has combining marks, so the lookup is worth skipping
      // outright rather than doing it for every cell on the screen.
      if (!combining_marks_.empty()) {
        auto marks = combining_marks_.find(CellIndex(x, y));
        if (marks != combining_marks_.end()) {
          out.Append(marks->second);
        }
      }
    }

    active->AppendTransitionTo(out, default_style, mode);
    out.Append("\n");
  }
}

auto Buffer::WriteTo(Filesystem::WriteFileRef file, ColorMode mode) const
    -> ErrorOr<Success, Filesystem::FdError> {
  // Sized for the few short lines a diagnostic renders to. A full screen with
  // color runs well past it and allocates once.
  llvm::SmallString<1024> bytes;
  Render(bytes, mode);
  return file.WriteCompleteBuffer(llvm::ArrayRef<std::byte>(
      reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()));
}

}  // namespace Carbon::Terminal

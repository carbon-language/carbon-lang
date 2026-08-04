// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <algorithm>
#include <array>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Unicode.h"

namespace Carbon::Terminal {

// Columns between tab stops, measured from where the text began rather than
// from the left edge of the buffer.
static constexpr int TabWidth = 8;

// The most bytes of combining marks kept on one cell. Text stacking more than
// this is either adversarial or already illegible, and keeping all of it would
// let one input character produce unbounded output.
static constexpr size_t MaxCombiningBytes = 32;

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

Buffer::Buffer(int width, Charset charset) : width_(width), metrics_(charset) {
  CARBON_CHECK(width > 0, "Buffer width must be positive, but was {0}.", width);
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

auto Buffer::EnsureWidth(int x) -> void {
  if (x < width_) {
    return;
  }
  // Growing by halves rather than to exactly what was asked keeps a row drawn
  // one symbol at a time from reflowing on every symbol.
  int width = std::max(x + 1, width_ + width_ / 2);

  // Rows are stored back to back, so every row but the first moves.
  int rows = height();
  llvm::SmallVector<Cell, 0> widened(static_cast<size_t>(rows) * width);
  for (int y : llvm::seq(rows)) {
    llvm::copy(
        llvm::ArrayRef(cells_).slice(static_cast<size_t>(y) * width_, width_),
        widened.begin() + static_cast<size_t>(y) * width);
  }
  cells_ = std::move(widened);

  // The marks are keyed by cell index, which is where the row moved it to.
  llvm::DenseMap<int, std::string> moved;
  moved.reserve(combining_marks_.size());
  for (auto& [index, marks] : combining_marks_) {
    moved.insert({index / width_ * width + index % width_, std::move(marks)});
  }
  combining_marks_ = std::move(moved);

  width_ = width;
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
  // Marks render into the column before them, so there must be one and it must
  // already have been drawn.
  if (x <= 0 || x > width_ || y < 0 || y >= height()) {
    return;
  }
  int base = x - 1;
  if (CellAt(base, y).is_continuation) {
    --base;
  }

  Utf8Storage storage;
  llvm::StringRef encoded = EncodeUtf8(symbol, storage);
  std::string& marks = combining_marks_[CellIndex(base, y)];
  if (marks.size() + encoded.size() > MaxCombiningBytes) {
    return;
  }
  marks.append(encoded.data(), encoded.size());
}

auto Buffer::DrawSymbol(int x, int y, char32_t symbol, const Style& style)
    -> DrawEnd {
  // A combining mark takes no column of its own, and renders into the one
  // before it instead.
  int width = metrics_.SymbolWidth(symbol);
  if (width == 0) {
    AttachCombiningMark(x, y, symbol);
    return {.x = x, .y = y};
  }
  symbol = metrics_.RenderedSymbol(symbol);

  if (x < 0 || y < 0) {
    return {.x = x + width, .y = y};
  }

  // A double-width symbol needs both its columns, since splitting one would
  // leave the terminal rendering half a character.
  EnsureWidth(x + width - 1);
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
  return {.x = x + width, .y = y};
}

auto Buffer::DrawLine(int x, int y, uint8_t directions, const Style& style)
    -> void {
  if (x < 0 || y < 0) {
    return;
  }
  CARBON_DCHECK(directions < Utf8LineGlyphs.size());
  EnsureWidth(x);
  EnsureRow(y);

  uint8_t existing = CellAt(x, y).lines;
  if (existing == 0) {
    // Whatever is here isn't a line. Clearing also removes either half of a
    // double-width symbol the cell was part of.
    ClearCells(x, y, 1);
  }

  Cell& cell = CellAt(x, y);
  cell.lines = existing | directions;
  cell.symbol = metrics_.charset() == Charset::Utf8
                    ? Utf8LineGlyphs[cell.lines]
                    : AsciiLineGlyphs[cell.lines];
  cell.style = style;
}

auto Buffer::DrawHorizontalLine(int x, int y, int length, const Style& style)
    -> DrawEnd {
  for (int i = 0; i < length; ++i) {
    // The ends of a line only connect inward, so a line meeting another at its
    // end forms a corner rather than a crossing. A one-cell line has no inward
    // direction and is drawn as a bare horizontal segment.
    uint8_t directions =
        length == 1 ? LineLeft | LineRight
                    : (i > 0 ? LineLeft : 0) | (i + 1 < length ? LineRight : 0);
    DrawLine(x + i, y, directions, style);
  }
  return {.x = x + std::max(length, 0), .y = y};
}

auto Buffer::DrawVerticalLine(int x, int y, int length, const Style& style)
    -> DrawEnd {
  for (int i = 0; i < length; ++i) {
    uint8_t directions =
        length == 1 ? LineUp | LineDown
                    : (i > 0 ? LineUp : 0) | (i + 1 < length ? LineDown : 0);
    DrawLine(x, y + i, directions, style);
  }
  return {.x = x, .y = y + std::max(length, 0)};
}

auto Buffer::DrawBox(int x, int y, int box_width, int box_height,
                     const Style& style) -> DrawEnd {
  if (box_width <= 0 || box_height <= 0) {
    return {.x = x, .y = y};
  }
  // A box with no interior is just the one line that bounds it.
  if (box_width == 1) {
    return DrawVerticalLine(x, y, box_height, style);
  }
  if (box_height == 1) {
    return DrawHorizontalLine(x, y, box_width, style);
  }

  DrawHorizontalLine(x, y, box_width, style);
  DrawHorizontalLine(x, y + box_height - 1, box_width, style);
  DrawVerticalLine(x, y, box_height, style);
  DrawVerticalLine(x + box_width - 1, y, box_height, style);
  return {.x = x + box_width, .y = y + box_height};
}

template <typename PlaceFn>
auto Buffer::WalkText(int x, int y, llvm::StringRef text, PlaceFn place) const
    -> DrawEnd {
  int cur_x = x;
  int cur_y = y;

  while (!text.empty()) {
    char32_t symbol = metrics_.TakeSymbol(text);
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
        place(cur_x, cur_y, U' ');
      }
      continue;
    }

    cur_x = place(cur_x, cur_y, symbol);
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawText(int x, int y, llvm::StringRef text, const Style& style)
    -> DrawEnd {
  return WalkText(x, y, text, [&](int cur_x, int cur_y, char32_t symbol) {
    return DrawSymbol(cur_x, cur_y, symbol, style).x;
  });
}

auto Buffer::MeasureText(int x, int y, llvm::StringRef text) const -> DrawEnd {
  return WalkText(x, y, text, [&](int cur_x, int /*cur_y*/, char32_t symbol) {
    return cur_x + metrics_.SymbolWidth(symbol);
  });
}

template <typename PlaceFn>
auto Buffer::WalkWrappedText(int x, int y, int max_width, llvm::StringRef text,
                             PlaceFn place) const -> DrawEnd {
  if (max_width <= 0) {
    return {.x = x, .y = y};
  }

  int cur_x = x;
  int cur_y = y;

  // What a row has room for is decided from the columns it has used rather
  // than from a column computed once from `max_width`, so that a width no row
  // reaches works like any other instead of overflowing the column it names.
  //
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

    if (IsWrapBreak(text.front())) {
      llvm::StringRef spaces = text.take_while(IsWrapBreak);
      text = text.drop_front(spaces.size());
      // Spaces are dropped wherever they would begin a row, which is what
      // keeps wrapped text aligned with where it started. Carriage returns are
      // never drawn, so CRLF line endings break exactly once.
      if (cur_x > x) {
        for (char space : spaces) {
          if (space == '\r' || cur_x - x >= max_width) {
            continue;
          }
          cur_x = place(cur_x, cur_y, U' ');
        }
      }
      continue;
    }

    llvm::StringRef word =
        text.take_until([](char c) { return c == '\n' || IsWrapBreak(c); });
    text = text.drop_front(word.size());

    // Move a word that doesn't fit down to the next row. If it doesn't fit
    // there either it overhangs rather than being broken, and the buffer grows
    // to hold it.
    if (cur_x > x && cur_x - x + metrics_.Width(word) > max_width) {
      cur_x = x;
      ++cur_y;
    }

    while (!word.empty()) {
      cur_x = place(cur_x, cur_y, metrics_.TakeSymbol(word));
    }
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawWrappedText(int x, int y, int max_width, llvm::StringRef text,
                             const Style& style) -> DrawEnd {
  return WalkWrappedText(x, y, max_width, text,
                         [&](int cur_x, int cur_y, char32_t symbol) {
                           return DrawSymbol(cur_x, cur_y, symbol, style).x;
                         });
}

auto Buffer::MeasureWrappedText(int x, int y, int max_width,
                                llvm::StringRef text) const -> DrawEnd {
  return WalkWrappedText(x, y, max_width, text,
                         [&](int cur_x, int /*cur_y*/, char32_t symbol) {
                           return cur_x + metrics_.SymbolWidth(symbol);
                         });
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
  Utf8Storage storage;

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

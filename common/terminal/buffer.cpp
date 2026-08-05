// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Unicode.h"

namespace Carbon::Terminal {

// The most bytes of combining marks kept on one cell. Text stacking more than
// this is either adversarial or already illegible, and keeping all of it would
// let a single column of output carry unbounded bytes.
static constexpr size_t MaxCombiningBytes = 32;

// Glyphs for every combination of line directions, indexed by the direction
// bits.
static constexpr std::array<char32_t, 16> Utf8LineGlyphs = {
    U'·',  // (none): a line between one center and itself, which is a point
    U'╴',  // left
    U'╶',  // right
    U'─',  // left, right
    U'╵',  // up
    U'╯',  // left, up
    U'╰',  // right, up
    U'┴',  // left, right, up
    U'╷',  // down
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
    U'+', U'-', U'-', U'-', U'|', U'+', U'+', U'+',
    U'|', U'+', U'+', U'+', U'|', U'+', U'+', U'+',
};

// Returns the next tab stop after `x` on a line whose stops are `tab_width`
// columns apart counting from `origin`, which `x` must not be left of.
static auto NextTabStop(int x, int origin, int tab_width) -> int {
  CARBON_DCHECK(x >= origin, "Column {0} is left of the origin {1}.", x,
                origin);
  return origin + ((x - origin) / tab_width + 1) * tab_width;
}

Buffer::Buffer(int columns, Charset charset, int tab_width)
    : columns_(columns),
      width_(columns),
      tab_width_(tab_width),
      metrics_(charset) {
  CARBON_CHECK(columns > 0 && columns <= MaxColumns,
               "Buffer width must be in [1, {0}], but was {1}.", MaxColumns,
               columns);
  CARBON_CHECK(tab_width > 0 && tab_width <= MaxTabWidth,
               "Tab width must be in [1, {0}], but was {1}.", MaxTabWidth,
               tab_width);
}

auto Buffer::height() const -> int {
  return static_cast<int>(cells_.size()) / width_;
}

auto Buffer::EnsureRow(int y) -> void {
  CARBON_CHECK(y >= 0 && y < MaxRows, "Row {0} is outside [0, {1}).", y,
               MaxRows);
  if (y < height()) {
    return;
  }
  // Rows are added at the end and nothing already in the grid moves, so this
  // asks for exactly the rows wanted and lets the vector amortize the growing.
  cells_.resize(static_cast<size_t>(y + 1) * width_);
}

auto Buffer::EnsureWidth(int x) -> void {
  CARBON_CHECK(x >= 0 && x < MaxColumns, "Column {0} is outside [0, {1}).", x,
               MaxColumns);
  if (x < width_) {
    return;
  }
  // Widening moves every row, so it grows by halves rather than to exactly what
  // was asked: a row drawn one code point at a time would otherwise copy the
  // whole grid on every one of them. Growth stops at the bound, which is what
  // holds the product of the two dimensions inside what a cell index can
  // represent.
  int width = std::min(std::max(x + 1, width_ + width_ / 2), MaxColumns);

  int rows = height();
  llvm::SmallVector<Cell, 0> new_cells(static_cast<size_t>(rows) * width);
  for (int y : llvm::seq(rows)) {
    llvm::copy(
        llvm::ArrayRef(cells_).slice(static_cast<size_t>(y) * width_, width_),
        new_cells.begin() + static_cast<size_t>(y) * width);
  }
  cells_ = std::move(new_cells);

  // A mark's key is a cell index, which depends on the width, so each is
  // recomputed for the new one.
  llvm::DenseMap<int, std::string> new_combining_marks;
  new_combining_marks.reserve(combining_marks_.size());
  for (auto& [index, marks] : combining_marks_) {
    new_combining_marks.insert(
        {index / width_ * width + index % width_, std::move(marks)});
  }
  combining_marks_ = std::move(new_combining_marks);

  width_ = width;
}

auto Buffer::ClearCells(int x, int y, int width) -> void {
  CARBON_CHECK(
      x >= 0 && width >= 0 && x + width <= width_ && y >= 0 && y < height(),
      "Clearing [{0}, {1}) of row {2} reaches outside the {3}x{4} cells the "
      "buffer holds.",
      x, x + width, y, width_, height());

  // A cleared range must not leave half of a double-width character behind, so
  // it extends over either half that crosses its edges.
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

auto Buffer::AttachCombiningMark(int x, int y, char32_t code_point) -> void {
  // A mark has nowhere to go when no cell precedes it, so it is dropped.
  if (x <= 0 || x > width_ || y < 0 || y >= height()) {
    return;
  }
  // The left half of a double-width character is never itself a continuation,
  // so stepping back from one always lands on a real character.
  int base = x - 1;
  if (CellAt(base, y).is_continuation) {
    --base;
  }
  CARBON_CHECK(base >= 0, "A continuation cell at column zero has no base.");

  Utf8Storage storage;
  llvm::StringRef encoded = EncodeUtf8(code_point, storage);
  std::string& marks = combining_marks_[CellIndex(base, y)];
  if (marks.size() + encoded.size() > MaxCombiningBytes) {
    return;
  }
  marks.append(encoded.data(), encoded.size());
}

auto Buffer::DrawCodePoint(int x, int y, char32_t code_point,
                           const Style& style) -> DrawEnd {
  CheckOrigin(x, y);
  return {.x = PlaceCodePoint(x, y, code_point, style), .y = y};
}

auto Buffer::PlaceCodePoint(int x, int y, char32_t code_point,
                            const Style& style) -> int {
  CARBON_DCHECK(x >= 0 && y >= 0,
                "Placing at ({0}, {1}), which no walk should reach.", x, y);

  int width = metrics_.CodePointWidth(code_point);
  if (width == 0) {
    AttachCombiningMark(x, y, code_point);
    return x;
  }
  code_point = metrics_.RenderedCodePoint(code_point);

  // Both bounds are reached by what the text holds rather than by where the
  // caller aimed -- a word overhanging the target width, or newlines running
  // past the rows a grid can index -- so past either one nothing is drawn and
  // the column still advances, which is what keeps measuring and drawing
  // answering the same thing. A double-width character needs both its columns,
  // so one that would only half fit is past the edge like any other: splitting
  // it would leave the terminal rendering half a character.
  if (y >= MaxRows || x > MaxColumns - width) {
    return x + width;
  }

  EnsureWidth(x + width - 1);
  EnsureRow(y);
  ClearCells(x, y, width);

  Cell& cell = CellAt(x, y);
  cell.code_point = code_point;
  cell.style = style;
  // Nothing is wider than two columns, so the second is the only continuation
  // there can be.
  if (width > 1) {
    Cell& continuation = CellAt(x + 1, y);
    continuation.style = style;
    continuation.is_continuation = true;
  }
  return x + width;
}

// Returns the glyphs a cell's directions are read from.
static auto LineGlyphs(Charset charset) -> const std::array<char32_t, 16>& {
  return charset == Charset::Utf8 ? Utf8LineGlyphs : AsciiLineGlyphs;
}

auto Buffer::DrawLine(int x, int y, uint8_t directions, const Style& style)
    -> void {
  CARBON_DCHECK(directions <= LineDirections,
                "Direction bits {0} name no glyph.", directions);
  EnsureWidth(x);
  EnsureRow(y);

  uint8_t existing = CellAt(x, y).lines;
  if (existing == 0) {
    // Whatever is here isn't a line. Clearing also removes either half of a
    // double-width character the cell was part of.
    ClearCells(x, y, 1);
  }

  Cell& cell = CellAt(x, y);
  cell.lines = existing | directions | LineCell;
  cell.code_point = LineGlyphs(metrics_.charset())[cell.lines & LineDirections];
  cell.style = style;
}

// Checks that a line of `length` starting at `position` stays within `limit`,
// which is the width for a horizontal line and `MaxRows` for a vertical one.
//
// Unlike text, a line has no reason to reach outside what it is being drawn
// into: nothing about it is unbreakable, and a layout that put one there
// computed the wrong extent.
static auto CheckLineFits(int position, int length, int limit) -> void {
  CARBON_CHECK(length >= 0 && position <= limit - length,
               "A line of {0} at {1} runs outside the {2} available to it.",
               length, position, limit);
}

auto Buffer::DrawHorizontalLine(int x, int y, int length, const Style& style,
                                LineEnd start, LineEnd end) -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(x, length, columns_);
  for (int i : llvm::seq(length)) {
    // A cell in the middle of the line is entered from one side and left by the
    // other. An end cell is only left towards the rest of the line, unless that
    // end runs out through the cell's own side.
    uint8_t directions =
        (i > 0 || start == LineEnd::Edge ? LineLeft : 0) |
        (i + 1 < length || end == LineEnd::Edge ? LineRight : 0);
    DrawLine(x + i, y, directions, style);
  }
  return {.x = x + length, .y = y};
}

auto Buffer::DrawVerticalLine(int x, int y, int length, const Style& style,
                              LineEnd start, LineEnd end) -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(y, length, MaxRows);
  for (int i : llvm::seq(length)) {
    uint8_t directions =
        (i > 0 || start == LineEnd::Edge ? LineUp : 0) |
        (i + 1 < length || end == LineEnd::Edge ? LineDown : 0);
    DrawLine(x, y + i, directions, style);
  }
  return {.x = x, .y = y + length};
}

auto Buffer::DrawBox(int x, int y, int box_width, int box_height,
                     const Style& style) -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(x, box_width, columns_);
  CheckLineFits(y, box_height, MaxRows);
  if (box_width == 0 || box_height == 0) {
    return {.x = x, .y = y};
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
  CheckOrigin(x, y);
  CheckTextSize(text);

  int cur_x = x;
  int cur_y = y;

  while (!text.empty()) {
    char32_t code_point = metrics_.TakeCodePoint(text);
    if (code_point == '\n') {
      cur_x = x;
      ++cur_y;
      continue;
    }
    if (code_point == '\r') {
      cur_x = x;
      continue;
    }
    if (code_point == '\t') {
      int stop = NextTabStop(cur_x, x, tab_width_);
      for (; cur_x < stop; ++cur_x) {
        place(cur_x, cur_y, U' ');
      }
      continue;
    }

    cur_x = place(cur_x, cur_y, code_point);
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawText(int x, int y, llvm::StringRef text, const Style& style)
    -> DrawEnd {
  return WalkText(x, y, text, [&](int cur_x, int cur_y, char32_t code_point) {
    return PlaceCodePoint(cur_x, cur_y, code_point, style);
  });
}

auto Buffer::MeasureText(int x, int y, llvm::StringRef text) const -> DrawEnd {
  return WalkText(x, y, text,
                  [&](int cur_x, int /*cur_y*/, char32_t code_point) {
                    return cur_x + metrics_.CodePointWidth(code_point);
                  });
}

template <typename PlaceFn>
auto Buffer::WalkWrappedText(int x, int y, int margin, int max_width,
                             llvm::StringRef text, PlaceFn place) const
    -> DrawEnd {
  CheckOrigin(x, y);
  CheckTextSize(text);
  CARBON_CHECK(margin >= 0 && margin <= x && max_width > 0 &&
                   x - margin < max_width && margin <= columns_ - max_width,
               "A block of {0} columns at {1} holding text from {2} does not "
               "fit the {3} columns a buffer covers.",
               max_width, margin, x, columns_);

  // The column a row runs out of room at. The block lies within the buffer's
  // width, so this is a column like any other rather than a sum that has to be
  // kept from overflowing.
  int limit = margin + max_width;

  int cur_x = x;
  int cur_y = y;
  // Whether the row being written was started by wrapping, rather than by the
  // text itself. Spaces are dropped at the start of a wrapped row and kept at
  // the start of one the text asked for, which is the difference between
  // indentation the caller wrote and indentation an accident of width would
  // otherwise introduce.
  bool wrapped_row = false;

  // Splitting on bytes is safe because every character text can break at is
  // ASCII, and UTF-8 never encodes anything else using an ASCII byte. Only
  // words are decoded; whitespace is handled a byte at a time.
  while (!text.empty()) {
    if (text.front() == '\n') {
      text = text.drop_front();
      cur_x = margin;
      ++cur_y;
      wrapped_row = false;
      continue;
    }

    if (IsWrapBreak(text.front())) {
      llvm::StringRef breaks = text.take_while(IsWrapBreak);
      text = text.drop_front(breaks.size());
      if (cur_x > margin || !wrapped_row) {
        for (char c : breaks) {
          if (c == '\r') {
            continue;
          }
          // Whitespace stops at the block's edge, leaving the word after it to
          // wrap.
          int next = std::min(
              c == '\t' ? NextTabStop(cur_x, margin, tab_width_) : cur_x + 1,
              limit);
          while (cur_x < next) {
            cur_x = place(cur_x, cur_y, U' ');
          }
        }
      }
      continue;
    }

    llvm::StringRef word =
        text.take_until([](char c) { return c == '\n' || IsWrapBreak(c); });
    text = text.drop_front(word.size());

    // Move a word that doesn't fit down to the next row, which minimizes the
    // overhang when it doesn't fit there either.
    if (cur_x > margin && cur_x + metrics_.Width(word) > limit) {
      cur_x = margin;
      ++cur_y;
      wrapped_row = true;
    }

    while (!word.empty()) {
      cur_x = place(cur_x, cur_y, metrics_.TakeCodePoint(word));
    }
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawWrappedText(int x, int y, int margin, int max_width,
                             llvm::StringRef text, const Style& style)
    -> DrawEnd {
  return WalkWrappedText(x, y, margin, max_width, text,
                         [&](int cur_x, int cur_y, char32_t code_point) {
                           return PlaceCodePoint(cur_x, cur_y, code_point,
                                                 style);
                         });
}

auto Buffer::MeasureWrappedText(int x, int y, int margin, int max_width,
                                llvm::StringRef text) const -> DrawEnd {
  return WalkWrappedText(x, y, margin, max_width, text,
                         [&](int cur_x, int /*cur_y*/, char32_t code_point) {
                           return cur_x + metrics_.CodePointWidth(code_point);
                         });
}

auto Buffer::LastVisibleColumn(int y, ColorMode mode) const -> int {
  // A style only paints a blank cell if it is rendered at all, so with color
  // off a blank cell is padding whatever style it carries.
  bool styles_render = mode != ColorMode::NoColor;
  for (int x = width_ - 1; x >= 0; --x) {
    const Cell& cell = CellAt(x, y);
    if (cell.is_continuation || cell.code_point != ' ' ||
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

  // The style a terminal starts in, and the one it is left in.
  const Style default_style;

  // Cells outlive this loop, so the active style is tracked by pointing at one
  // rather than copying a whole style per cell. It carries across rows: a style
  // is usually still in use on the row below, and turning it off and back on
  // costs a reset and a fresh start for nothing.
  const Style* active = &default_style;

  int rows = height();
  for (int y = 0; y < rows; ++y) {
    int last = LastVisibleColumn(y, mode);
    for (int x = 0; x <= last; ++x) {
      const Cell& cell = CellAt(x, y);
      if (cell.is_continuation) {
        continue;
      }

      active->AppendTransitionTo(out, cell.style, mode);
      active = &cell.style;
      out.Append(EncodeUtf8(cell.code_point, storage));

      // Almost nothing has combining marks, so the lookup is worth skipping
      // outright rather than doing it for every cell on the screen.
      if (!combining_marks_.empty()) {
        auto marks = combining_marks_.find(CellIndex(x, y));
        if (marks != combining_marks_.end()) {
          out.Append(marks->second);
        }
      }
    }

    // A style is turned off before the newline in two cases. On the last row,
    // so that nothing is left set for whatever is printed after this and the
    // escape that turns it off still falls inside the rendering. And whenever
    // it paints where there is no glyph, because a terminal fills the rest of
    // the row with the background it is in when the row ends, so leaving one
    // set would paint a stripe out to the right edge that nothing asked for.
    if (y + 1 == rows || active->IsVisibleOnBlank()) {
      active->AppendTransitionTo(out, default_style, mode);
      active = &default_style;
    }
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

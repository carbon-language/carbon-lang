// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/terminal/buffer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"

namespace Carbon::Terminal {

// The most bytes of combining marks kept on one cell. Text stacking more than
// this is either adversarial or already illegible, and keeping all of it would
// let a single column of output carry unbounded bytes.
static constexpr size_t MaxCombiningBytes = 32;

// Glyphs for every combination of line directions and weights, indexed by the
// heavy bits and then the direction bits.
//
// The first row is the all-light set, whose corners are the rounded forms; a
// corner with any heavy line is square, since Unicode has no rounded heavy
// corner. The other rows match the Unicode character names, which list the
// weight of the line in each direction. `DrawLine` never sets a heavy bit
// without its direction bit, and those entries repeat the glyph with the
// direction present.
static constexpr std::array<std::u32string_view, 16> Utf8LineGlyphs = {
    U"·╴╶─╵╯╰┴╷╮╭┬│┤├┼",  // All light.
    U"╸╸╾╾┙┙┵┵┑┑┭┭┥┥┽┽",  // Heavy left.
    U"╺╼╺╼┕┶┕┶┍┮┍┮┝┾┝┾",  // Heavy right.
    U"━━━━┷┷┷┷┯┯┯┯┿┿┿┿",  // Heavy left and right.
    U"╹┚┖┸╹┚┖┸╿┦┞╀╿┦┞╀",  // Heavy up.
    U"┛┛┹┹┛┛┹┹┩┩╃╃┩┩╃╃",  // Heavy left and up.
    U"┗┺┗┺┗┺┗┺┡╄┡╄┡╄┡╄",  // Heavy right and up.
    U"┻┻┻┻┻┻┻┻╇╇╇╇╇╇╇╇",  // Heavy left, right, and up.
    U"╻┒┎┰╽┧┟╁╻┒┎┰╽┧┟╁",  // Heavy down.
    U"┓┓┱┱┪┪╅╅┓┓┱┱┪┪╅╅",  // Heavy left and down.
    U"┏┲┏┲┢╆┢╆┏┲┏┲┢╆┢╆",  // Heavy right and down.
    U"┳┳┳┳╈╈╈╈┳┳┳┳╈╈╈╈",  // Heavy left, right, and down.
    U"┃┨┠╂┃┨┠╂┃┨┠╂┃┨┠╂",  // Heavy up and down.
    U"┫┫╉╉┫┫╉╉┫┫╉╉┫┫╉╉",  // Heavy left, up, and down.
    U"┣╊┣╊┣╊┣╊┣╊┣╊┣╊┣╊",  // Heavy right, up, and down.
    U"╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋╋",  // All heavy.
};
// Every row has a glyph for each set of direction bits.
static_assert(std::all_of(Utf8LineGlyphs.begin(), Utf8LineGlyphs.end(),
                          [](std::u32string_view glyphs) {
                            return glyphs.size() == 16;
                          }));

// The ASCII glyphs, which use `+` only for a crossing:
//
// - A line running through horizontally is `-`, vertically `|`, and both ways
//   `+`.
// - A corner is `.` where its line leaves downward and `'` where it arrives
//   from above, which is where those characters sit in their cells.
// - A tee where a line leaves a vertical line sideways, `├` or `┤`, is `|`,
//   and the dash drawn beside it shows the line leaving. A tee where a line
//   leaves a horizontal line upward or downward, `┴` or `┬`, is the corner
//   glyph for that direction: the dashes either side show the horizontal line,
//   and this cell must show the line leaving it, because the rest of that line
//   can be rows away with only gaps between. `+` would look like a crossing.
// - A point, a line between one center and itself, is `.`.
//
// What a diagnostic draws is then still told apart: the line closing a snippet
// from the one separating two, and the line opening a diagnostic from the one
// continuing it.
//
// A cell with a heavy line leaving it is `^` instead, under the rule
// `LineWeight` describes. That covers the top cell of a heavy vertical line, so
// the line is `^` above `|`.
static constexpr std::array<char32_t, 16> AsciiLineGlyphs = {
    U'.',   // None: a point.
    U'-',   // Left.
    U'-',   // Right.
    U'-',   // Left and right.
    U'|',   // Up.
    U'\'',  // Left and up.
    U'\'',  // Right and up.
    U'\'',  // Left, right, and up.
    U'|',   // Down.
    U'.',   // Left and down.
    U'.',   // Right and down.
    U'.',   // Left, right, and down.
    U'|',   // Up and down.
    U'|',   // Left, up, and down.
    U'|',   // Right, up, and down.
    U'+',   // Left, right, up, and down.
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

auto Buffer::EnsureColumn(int x) -> void {
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
  CheckTextOrigin(x, y);
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

  EnsureColumn(x + width - 1);
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

auto Buffer::DrawLine(int x, int y, uint8_t directions, LineWeight weight,
                      const Style& style) -> void {
  CARBON_DCHECK(directions <= LineDirections,
                "Direction bits {0} name no glyph.", directions);
  EnsureColumn(x);
  EnsureRow(y);

  uint8_t existing = CellAt(x, y).lines;
  if (existing == 0) {
    // Whatever is here isn't a line. Clearing also removes either half of a
    // double-width character the cell was part of.
    ClearCells(x, y, 1);
  }

  Cell& cell = CellAt(x, y);
  cell.lines = existing | directions | LineCell;
  cell.heavy_lines &= ~directions;
  if (weight == LineWeight::Heavy) {
    cell.heavy_lines |= directions;
  }
  uint8_t cell_directions = cell.lines & LineDirections;
  if (metrics_.charset() == Charset::Utf8) {
    cell.code_point = Utf8LineGlyphs[cell.heavy_lines][cell_directions];
  } else if (cell.heavy_lines != 0 && (cell_directions & LineUp) == 0 &&
             ((cell_directions & LineDown) == 0 ||
              (cell.heavy_lines & LineDown) != 0)) {
    cell.code_point = U'^';
  } else {
    cell.code_point = AsciiLineGlyphs[cell_directions];
  }
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
                                LineEnd start, LineEnd end, LineWeight weight)
    -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(x, length, columns_);
  for (int i : llvm::seq(length)) {
    // A cell in the middle of the line is entered from one side and left by the
    // other. An end cell is only left towards the rest of the line, unless that
    // end runs out through the cell's own side.
    uint8_t directions =
        (i > 0 || start == LineEnd::Edge ? LineLeft : 0) |
        (i + 1 < length || end == LineEnd::Edge ? LineRight : 0);
    DrawLine(x + i, y, directions, weight, style);
  }
  return {.x = x + length, .y = y};
}

auto Buffer::DrawVerticalLine(int x, int y, int length, const Style& style,
                              LineEnd start, LineEnd end, LineWeight weight)
    -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(y, length, MaxRows);
  for (int i : llvm::seq(length)) {
    uint8_t directions =
        (i > 0 || start == LineEnd::Edge ? LineUp : 0) |
        (i + 1 < length || end == LineEnd::Edge ? LineDown : 0);
    DrawLine(x, y + i, directions, weight, style);
  }
  return {.x = x, .y = y + length};
}

auto Buffer::HoldsText(int x, int y) const -> bool {
  if (x < 0 || x >= width_ || y >= height()) {
    return false;
  }
  const Cell& cell = CellAt(x, y);
  return (cell.lines & LineCell) == 0 &&
         (cell.code_point != U' ' || cell.is_continuation ||
          (!combining_marks_.empty() &&
           combining_marks_.contains(CellIndex(x, y))));
}

auto Buffer::DrawCrossingVerticalLine(int x, int y, int length,
                                      const Style& style, LineEnd start)
    -> DrawEnd {
  CheckOrigin(x, y);
  CheckLineFits(y, length, MaxRows);

  for (int i : llvm::seq(length)) {
    int cell_y = y + i;
    bool is_center_start = i == 0 && start == LineEnd::Center;
    uint8_t directions = is_center_start ? LineDown : (LineUp | LineDown);
    if (cell_y >= height()) {
      // Rows below the buffer hold nothing drawn earlier.
      DrawLine(x, cell_y, directions, LineWeight::Light, style);
      continue;
    }
    const Cell& cell = CellAt(x, cell_y);
    if (is_center_start) {
      // A line already leaving this cell downward is left as it is, keeping
      // its weight. Otherwise this line joins the cell.
      if ((cell.lines & LineDown) == 0) {
        DrawLine(x, cell_y, LineDown, LineWeight::Light, style);
      }
      continue;
    }
    if ((cell.lines & LineCell) != 0) {
      // An earlier line keeps its cell: a crossed line stays unbroken, and a
      // line already running down this column keeps its weight.
      continue;
    }
    // Skip text, and blank cells beside text, where a line could be mistaken
    // for part of a word.
    if (HoldsText(x, cell_y) || HoldsText(x - 1, cell_y) ||
        HoldsText(x + 1, cell_y)) {
      continue;
    }
    DrawLine(x, cell_y, directions, LineWeight::Light, style);
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
auto Buffer::WalkText(int x, int y, int margin, llvm::StringRef text,
                      PlaceFn place) const -> DrawEnd {
  CheckTextSize(text);
  CARBON_CHECK(margin >= 0 && margin <= x && y >= 0 && y < MaxRows,
               "Text at ({0}, {1}) with a margin of {2} is outside the {3} "
               "rows a buffer covers, or left of its margin.",
               x, y, margin, MaxRows);

  int cur_x = x;
  int cur_y = y;

  while (!text.empty()) {
    char32_t code_point = metrics_.TakeCodePoint(text);
    if (code_point == '\n') {
      cur_x = margin;
      ++cur_y;
      continue;
    }
    if (code_point == '\r') {
      cur_x = margin;
      continue;
    }
    if (code_point == '\t') {
      int stop = NextTabStop(cur_x, margin, tab_width_);
      for (; cur_x < stop; ++cur_x) {
        place(cur_x, cur_y, U' ');
      }
      continue;
    }

    cur_x = place(cur_x, cur_y, code_point);
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawText(int x, int y, int margin, llvm::StringRef text,
                      const Style& style) -> DrawEnd {
  return WalkText(x, y, margin, text,
                  [&](int cur_x, int cur_y, char32_t code_point) {
                    return PlaceCodePoint(cur_x, cur_y, code_point, style);
                  });
}

auto Buffer::MeasureText(int x, int y, int margin, llvm::StringRef text) const
    -> DrawEnd {
  return WalkText(x, y, margin, text,
                  [&](int cur_x, int /*cur_y*/, char32_t code_point) {
                    return cur_x + metrics_.CodePointWidth(code_point);
                  });
}

// Returns whether wrapped text can be broken at `c`.
//
// Carriage returns count so that a CRLF ending is whitespace rather than part
// of the word before it; what becomes of the `\r` is then up to the drawing.
static constexpr auto IsWrapBreak(char c) -> bool {
  return c == ' ' || c == '\t' || c == '\r';
}

// Removes the word at the front of `text` and returns it: everything up to the
// next break or newline, except that with `keep_backtick_quotes_whole`, a pair
// of backticks on one line keeps everything between them in the word. When
// `text` starts with a break or a newline, returns an empty word and leaves
// `text` unchanged, so the caller has to step past that character itself.
//
// With `IsWrapBreak` this is the one definition of where wrapping may
// introduce a break, so that measuring what text wraps into and drawing it
// wrapped agree about it.
static auto TakeWord(llvm::StringRef& text, bool keep_backtick_quotes_whole)
    -> llvm::StringRef {
  size_t end = 0;
  while (end < text.size()) {
    char c = text[end];
    if (c == '\n' || IsWrapBreak(c)) {
      break;
    }
    if (keep_backtick_quotes_whole && c == '`') {
      size_t close = text.find('`', end + 1);
      if (close != llvm::StringRef::npos &&
          text.slice(end + 1, close).find_first_of("\t\n\r") ==
              llvm::StringRef::npos) {
        end = close + 1;
        continue;
      }
    }
    ++end;
  }
  llvm::StringRef word = text.take_front(end);
  text = text.drop_front(end);
  return word;
}

template <typename PlaceFn>
auto Buffer::WalkWrappedText(int x, int y, int margin, int max_width,
                             llvm::StringRef text,
                             bool keep_backtick_quotes_whole,
                             PlaceFn place) const -> DrawEnd {
  CheckTextSize(text);
  // The block runs from the margin to `margin + max_width`, lies within the
  // buffer, and holds the column the text starts in, which is every bound on
  // the three of them read in one order.
  CARBON_CHECK(llvm::is_sorted(std::array{0, margin, x, x + 1,
                                          margin + max_width, columns_}) &&
                   y >= 0 && y < MaxRows,
               "A block of {0} columns at {1} holding text from ({2}, {3}) "
               "does not fit the {4} columns and {5} rows a buffer covers.",
               max_width, margin, x, y, columns_, MaxRows);

  // The column a row runs out of room at. The block lies within the buffer's
  // width, so this is a column like any other rather than a sum that has to be
  // kept from overflowing.
  int limit = margin + max_width;

  int cur_x = x;
  int cur_y = y;

  // Splitting on bytes is safe because every character text can break at is
  // ASCII, and UTF-8 never encodes anything else using an ASCII byte. Only
  // words are decoded; whitespace is handled a byte at a time.
  while (!text.empty()) {
    if (text.front() == '\n') {
      text = text.drop_front();
      cur_x = margin;
      ++cur_y;
      continue;
    }

    if (IsWrapBreak(text.front())) {
      llvm::StringRef breaks = text.take_while(IsWrapBreak);
      text = text.drop_front(breaks.size());
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

      // A combining mark renders into the column before it, so one following
      // whitespace belongs to that whitespace and goes with it. Left to begin
      // the next word, it would move to another row whenever that word wrapped
      // and attach to whatever preceded it there.
      while (!text.empty()) {
        llvm::StringRef rest = text;
        char32_t code_point = metrics_.TakeCodePoint(rest);
        if (metrics_.CodePointWidth(code_point) != 0) {
          break;
        }
        text = rest;
        cur_x = place(cur_x, cur_y, code_point);
      }
      continue;
    }

    llvm::StringRef word = TakeWord(text, keep_backtick_quotes_whole);

    // Move a word that doesn't fit down to the next row, which minimizes the
    // overhang when it doesn't fit there either. The word is drawn into the row
    // this starts before anything else can reach it, so a wrapped row begins at
    // the margin rather than with the whitespace the wrap came after.
    if (cur_x > margin && cur_x + metrics_.Width(word) > limit) {
      cur_x = margin;
      ++cur_y;
    }

    while (!word.empty()) {
      cur_x = place(cur_x, cur_y, metrics_.TakeCodePoint(word));
    }
  }

  return {.x = cur_x, .y = cur_y};
}

auto Buffer::DrawWrappedText(int x, int y, int margin, int max_width,
                             llvm::StringRef text, const Style& style,
                             bool keep_backtick_quotes_whole) -> DrawEnd {
  return WalkWrappedText(
      x, y, margin, max_width, text, keep_backtick_quotes_whole,
      [&](int cur_x, int cur_y, char32_t code_point) {
        return PlaceCodePoint(cur_x, cur_y, code_point, style);
      });
}

auto Buffer::MeasureWrappedText(int x, int y, int margin, int max_width,
                                llvm::StringRef text,
                                bool keep_backtick_quotes_whole) const
    -> DrawEnd {
  return WalkWrappedText(x, y, margin, max_width, text,
                         keep_backtick_quotes_whole,
                         [&](int cur_x, int /*cur_y*/, char32_t code_point) {
                           return cur_x + metrics_.CodePointWidth(code_point);
                         });
}

auto Buffer::MeasureWrapWidth(llvm::StringRef text,
                              bool keep_backtick_quotes_whole) const -> int {
  int width = 0;
  while (!text.empty()) {
    llvm::StringRef word = TakeWord(text, keep_backtick_quotes_whole);
    width = std::max(width, metrics_.Width(word));
    if (word.empty()) {
      text = text.drop_front();
    }
  }
  return width;
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

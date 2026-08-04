// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_BUFFER_H_
#define CARBON_COMMON_TERMINAL_BUFFER_H_

#include <cstdint>
#include <string>

#include "common/filesystem.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/color.h"
#include "common/terminal/metrics.h"
#include "common/terminal/output_buffer_ref.h"
#include "common/terminal/style.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// A grid of styled cells staged for rendering to a terminal.
//
// Coordinates are 0-based with (0, 0) at the top left, `x` counting terminal
// columns and `y` counting rows.
//
// A buffer renders once, top to bottom, the way a compiler writes diagnostics.
// There is no cursor addressing and nothing is ever redrawn, so a rendered
// buffer is just as valid in a file or a pipe as on a terminal.
//
// Staging into a grid lets layout position content directly, rather than
// interleaving text, padding, and escape sequences as it goes. That separation
// is what makes the two hard parts tractable: escape sequences are minimized
// once, in `Render`, and the drawing APIs reason about columns on screen rather
// than bytes in a stream.
//
// Which bytes make up a column depends on the charset, and the buffer handles
// that rather than leaving it to callers, because getting it wrong misaligns
// everything downstream of it:
//
// - Under `Charset::Ascii` no UTF-8 processing happens at all. Every byte is
//   one column, exactly as a terminal decoding some single-byte encoding will
//   treat it, and bytes outside printable ASCII are replaced with `?` because
//   there is no telling what such a terminal would draw for them.
// - Under `Charset::Utf8` bytes are decoded as UTF-8. Double-width characters
//   occupy both of the columns they will really take, and drawing over either
//   column erases the whole character instead of leaving half of one behind.
//   Combining marks render into the column before them, so a base character
//   and its marks stay in one cell. Carbon source is in Unicode normalization
//   form C, which still spells out marks for characters that have no
//   precomposed form, so this comes up in ordinary input. Anything with no
//   printable rendering, including invalid UTF-8, becomes U+FFFD.
//
// A buffer grows to hold whatever is drawn into it, in both directions.
// Nothing is clipped, so a caller never has to work out how wide its output
// will be before it starts producing it -- a measuring pass that mirrors the
// drawing pass, and that is wrong whenever the two disagree about a string.
// What to draw is still the caller's to decide: a terminal's width bounds how
// much source a snippet shows, not how much the grid can hold.
class Buffer {
 public:
  // Where a drawing ended: the row it ended on, and the column after its last
  // symbol on that row.
  //
  // Everything that draws returns one, so that a caller placing something
  // after a drawing advances from this rather than measuring the same text a
  // second time. The `Measure` operations return one too, and answer for text
  // that hasn't been drawn yet what drawing it would answer.
  struct DrawEnd {
    int x;
    int y;

    friend auto operator==(DrawEnd lhs, DrawEnd rhs) -> bool = default;
  };

  // Constructs an empty buffer holding `charset`.
  explicit Buffer(Charset charset) : Buffer(1, charset) {}

  // Constructs an empty buffer starting `width` columns wide, which must be
  // positive. The width is a starting size rather than a bound; it grows to
  // fit. Passing one saves the regrowing, and passing none costs only that.
  Buffer(int width, Charset charset);

  // Constructs an empty buffer holding `capabilities`'s charset, starting at
  // its width when there is one to start from.
  explicit Buffer(const Capabilities& capabilities)
      : Buffer(capabilities.columns.value_or(1), capabilities.charset) {}

  // Returns the columns the buffer currently holds, which is at least as many
  // as anything drawn into it occupies.
  auto width() const -> int { return width_; }

  // Returns the number of rows drawn into so far.
  auto height() const -> int;

  auto charset() const -> Charset { return metrics_.charset(); }

  // Returns how text is measured for this buffer's charset.
  //
  // The buffer lays its cells out with this, so a caller deciding where to put
  // something asks the same thing the drawing will. Most such questions are
  // about text rather than about a drawing -- how wide a word is, where to cut
  // a line -- and want this rather than a buffer.
  auto metrics() const -> Metrics { return metrics_; }

  // Returns where `DrawText` would end for these arguments, without drawing.
  //
  // Measuring and drawing walk the text with the same code, differing only in
  // whether they write a cell, so a layout decision made from this can't
  // disagree with what drawing then does. Measurement written separately would
  // have to agree about every string, and the ones it is most likely to get
  // wrong -- a tab, a newline, a double-width character -- are the ones whose
  // being wrong is hardest to see.
  //
  // This is for text that a tab or a newline makes positional. Text without
  // either is as wide wherever it is drawn, and `Metrics::Width` answers for
  // it without a buffer to draw into.
  auto MeasureText(int x, int y, llvm::StringRef text) const -> DrawEnd;

  // Returns where `DrawWrappedText` would end for these arguments, without
  // drawing.
  auto MeasureWrappedText(int x, int y, int max_width,
                          llvm::StringRef text) const -> DrawEnd;

  // Draws `symbol` at (x, y), growing the buffer as needed to reach it.
  //
  // Returns the column after it, which is `x` again for a combining mark since
  // one renders into the column before it. A symbol before the buffer's origin
  // still reports the columns it would have taken, so that text walking a row
  // stays in step.
  auto DrawSymbol(int x, int y, char32_t symbol, const Style& style) -> DrawEnd;

  // Draws a horizontal line of `length` columns starting at (x, y).
  //
  // Lines join wherever they overlap: a cell records which directions lines
  // leave it in, and its glyph follows from those bits alone, so crossings,
  // corners, and tees all appear without being asked for and whatever order
  // the lines were drawn in. This is the only way to produce a junction, and
  // it suffices because a junction in real line art always has the lines that
  // imply it running through it. Only line drawing records directions, so text
  // containing `-` or `+` is never redrawn as line art.
  //
  // A cell's style is whatever was drawn there last, so crossing lines of
  // different styles do depend on order.
  auto DrawHorizontalLine(int x, int y, int length, const Style& style)
      -> DrawEnd;

  // Draws a vertical line of `length` rows starting at (x, y). Returns the row
  // after it, in the column it ran down.
  auto DrawVerticalLine(int x, int y, int length, const Style& style)
      -> DrawEnd;

  // Draws the outline of a box with its top-left corner at (x, y). Its corners
  // come out of the four sides meeting. A box with no interior is the single
  // line that bounds it, and one with no extent draws nothing.
  auto DrawBox(int x, int y, int box_width, int box_height, const Style& style)
      -> DrawEnd;

  // Draws `text` starting at (x, y).
  //
  // Newlines return to column `x` on the next row, carriage returns to column
  // `x` on the same row, and tabs advance to the next eight-column stop
  // measured from `x`. Returns where it ended, which for text with a newline
  // in it is on a later row than it started.
  auto DrawText(int x, int y, llvm::StringRef text, const Style& style)
      -> DrawEnd;

  // Draws `text` starting at (x, y), wrapping it within `max_width` columns.
  //
  // Wrapping breaks at ASCII spaces and tabs, and only there. A word too long
  // for a row of its own is moved down to one and then overhangs it: breaking
  // a path or a type name costs a reader more than the overhang does, and the
  // buffer grows to hold what overhangs. Spaces are dropped wherever they
  // would begin a row, including at the start of the text, so wrapped text
  // stays aligned to `x`.
  //
  // Newlines break the line as they do in `DrawText`, and carriage returns are
  // dropped so that CRLF endings break exactly once. Tabs are break
  // opportunities that render as a single space rather than advancing to a tab
  // stop, unlike in `DrawText`, because a tab stop is measured from the start
  // of a line and wrapped text has no fixed one.
  //
  // Any positive `max_width` works, however large, and one no row can reach
  // wraps nothing. That is what a caller with no width to fit passes, rather
  // than there being a second way to draw for having none.
  //
  // Returns where it ended, which is (x, y) when `max_width` isn't positive,
  // in which case nothing is drawn.
  auto DrawWrappedText(int x, int y, int max_width, llvm::StringRef text,
                       const Style& style) -> DrawEnd;

  // Renders the grid, appending the bytes that draw it to `out`.
  //
  // Each row ends in a newline, with trailing blank cells dropped so output
  // carries no invisible padding, and with any style turned off so it can't
  // bleed into whatever is printed next. Color is chosen here rather than at
  // construction because it affects only how cells are serialized, while the
  // charset decides how content is laid out into them.
  auto Render(OutputBufferRef out, ColorMode mode) const -> void;

  // Renders the grid and writes it to `file`.
  //
  // The whole grid goes out in one `write` where the destination accepts it,
  // which is what gives the output whatever atomicity the descriptor offers
  // against other writers: a terminal or a pipe interleaves at write
  // boundaries, so one call per rendered buffer is the most that can be had
  // without a lock.
  auto WriteTo(Filesystem::WriteFileRef file, ColorMode mode) const
      -> ErrorOr<Success, Filesystem::FdError>;

 private:
  // The directions in which drawn lines leave a cell. A cell's glyph is a
  // function of these bits alone.
  enum LineDirection : uint8_t {
    LineLeft = 1 << 0,
    LineRight = 1 << 1,
    LineUp = 1 << 2,
    LineDown = 1 << 3,
  };

  struct Cell {
    // The code point rendered here. For a cell with `lines` set, this is
    // derived from those bits and the charset.
    char32_t symbol = ' ';

    Style style;

    // Which directions drawn lines leave this cell in, or zero for a cell
    // holding text.
    uint8_t lines = 0;

    // Whether this cell is the right half of a double-width symbol, and so
    // renders nothing of its own.
    bool is_continuation = false;
  };

  auto CellIndex(int x, int y) const -> int { return y * width_ + x; }
  auto CellAt(int x, int y) -> Cell& { return cells_[CellIndex(x, y)]; }
  auto CellAt(int x, int y) const -> const Cell& {
    return cells_[CellIndex(x, y)];
  }

  // The walks behind the text operations, over which drawing and measuring are
  // the same code. `place` is called with each symbol and where it goes, and
  // returns the column after it: `DrawSymbol` when drawing, and the width alone
  // when measuring.
  template <typename PlaceFn>
  auto WalkText(int x, int y, llvm::StringRef text, PlaceFn place) const
      -> DrawEnd;
  template <typename PlaceFn>
  auto WalkWrappedText(int x, int y, int max_width, llvm::StringRef text,
                       PlaceFn place) const -> DrawEnd;

  // Adds rows until row `y` exists.
  auto EnsureRow(int y) -> void;

  // Widens the buffer until column `x` exists, reflowing the rows it already
  // holds, which are stored back to back.
  auto EnsureWidth(int x) -> void;

  // Resets the cells in row `y` spanning columns [x, x + width), along with
  // either half of a double-width symbol that straddles the range's edges.
  auto ClearCells(int x, int y, int width) -> void;

  // Appends `symbol` to the marks rendered with the cell before column `x`.
  auto AttachCombiningMark(int x, int y, char32_t symbol) -> void;

  // Adds `directions` to the lines through (x, y) and updates its glyph.
  auto DrawLine(int x, int y, uint8_t directions, const Style& style) -> void;

  // Returns the last column in row `y` that renders anything under `mode`, or
  // -1 when the row renders nothing.
  auto LastVisibleColumn(int y, ColorMode mode) const -> int;

  int width_;
  Metrics metrics_;

  llvm::SmallVector<Cell, 0> cells_;

  // Combining marks, as UTF-8, for the few cells that have any, keyed by cell
  // index. Kept out of `Cell` so that the common case of no marks costs
  // nothing per cell. Always empty under `Charset::Ascii`.
  llvm::DenseMap<int, std::string> combining_marks_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_BUFFER_H_

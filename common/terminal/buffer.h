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
#include "common/terminal/output_buffer_ref.h"
#include "common/terminal/style.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Terminal {

// A grid of styled cells staged for rendering to a terminal.
//
// Coordinates are 0-based with (0, 0) at the top left, `x` counting terminal
// columns and `y` counting rows. Width is fixed at construction to match the
// terminal, and drawing past it clips. Height grows to fit whatever is drawn,
// so layout code doesn't need to know how tall its output is before producing
// it.
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
// A buffer grows to hold whatever is drawn into it, in both directions.
// Nothing is clipped, so a caller never has to work out how wide its output
// will be before it starts producing it -- a measuring pass that mirrors the
// drawing pass, and that is wrong whenever the two disagree about a string.
// What to draw is still the caller's to decide: a terminal's width bounds how
// much source a snippet shows, not how much the grid can hold.
class Buffer {
 public:
  // Constructs an empty buffer holding `charset`.
  explicit Buffer(Charset charset) : Buffer(1, charset) {}

  // Constructs an empty buffer starting `width` columns wide, which must be
  // positive. The width is a starting size rather than a bound; it grows to
  // fit. Passing one saves the regrowing, and passing none costs only that.
  Buffer(int width, Charset charset);

  // Constructs an empty buffer starting at `capabilities`'s width and holding
  // its charset.
  explicit Buffer(const Capabilities& capabilities)
      : Buffer(capabilities.columns, capabilities.charset) {}

  // Returns the columns the buffer currently holds, which is at least as many
  // as anything drawn into it occupies.
  auto width() const -> int { return width_; }

  // Returns the number of rows drawn into so far.
  auto height() const -> int;

  auto charset() const -> Charset { return charset_; }

  // Returns the columns `text` occupies when drawn into this buffer.
  //
  // Newlines and tabs are not interpreted; they count as one column each, the
  // same as any other character with no rendering of its own.
  auto MeasureWidth(llvm::StringRef text) const -> int;

  // Draws `symbol` at (x, y), adding rows as needed to reach `y`.
  //
  // Returns the columns it occupies, which is what a caller walking across a
  // row should advance by. That is zero only for a combining mark, and is
  // otherwise the symbol's width whether or not it was actually drawn, so that
  // clipping a symbol doesn't shift what comes after it.
  auto DrawSymbol(int x, int y, char32_t symbol, const Style& style) -> int;

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
  auto DrawHorizontalLine(int x, int y, int length, const Style& style) -> void;

  // Draws a vertical line of `length` rows starting at (x, y).
  auto DrawVerticalLine(int x, int y, int length, const Style& style) -> void;

  // Draws the outline of a box with its top-left corner at (x, y). Its corners
  // come out of the four sides meeting. A box with no interior is the single
  // line that bounds it, and one with no extent draws nothing.
  auto DrawBox(int x, int y, int box_width, int box_height, const Style& style)
      -> void;

  // Draws `text` starting at (x, y).
  //
  // Newlines return to column `x` on the next row, carriage returns to column
  // `x` on the same row, and tabs advance to the next eight-column stop
  // measured from `x`. Returns the number of rows spanned, counting the row
  // the text ends on.
  auto DrawText(int x, int y, llvm::StringRef text, const Style& style) -> int;

  // Draws `text` starting at (x, y), wrapping it within `max_width` columns.
  //
  // Wrapping breaks at ASCII spaces and tabs. A word too long for a row of its
  // own is moved down and then broken as it is drawn, and a symbol wider than
  // `max_width` is dropped, since no row could hold it. Spaces are dropped
  // wherever they would begin a row, including at the start of the text, so
  // wrapped text stays aligned to `x`.
  //
  // Newlines break the line as they do in `DrawText`, and carriage returns are
  // dropped so that CRLF endings break exactly once. Tabs are break
  // opportunities that render as a single space rather than advancing to a tab
  // stop, unlike in `DrawText`, because a tab stop is measured from the start
  // of a line and wrapped text has no fixed one.
  //
  // Returns the number of rows spanned, or zero when `max_width` isn't
  // positive, in which case nothing is drawn.
  auto DrawWrappedText(int x, int y, int max_width, llvm::StringRef text,
                       const Style& style) -> int;

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

  // Removes the next symbol from `text`, which must not be empty, and returns
  // it: one byte under `Charset::Ascii`, and one decoded code point under
  // `Charset::Utf8`.
  auto TakeSymbol(llvm::StringRef& text) const -> char32_t;

  // Returns the columns `symbol` occupies once drawn, which is what
  // `DrawSymbol` returns for it.
  auto SymbolWidth(char32_t symbol) const -> int;

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
  Charset charset_;

  llvm::SmallVector<Cell, 0> cells_;

  // Combining marks, as UTF-8, for the few cells that have any, keyed by cell
  // index. Kept out of `Cell` so that the common case of no marks costs
  // nothing per cell. Always empty under `Charset::Ascii`.
  llvm::DenseMap<int, std::string> combining_marks_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_BUFFER_H_

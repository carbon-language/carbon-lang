// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TERMINAL_BUFFER_H_
#define CARBON_COMMON_TERMINAL_BUFFER_H_

#include <algorithm>
#include <cstdint>
#include <string>

#include "common/check.h"
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

// Where a line stops within the cell at one of its ends.
//
// A line runs between points, and in a grid of cells the two points it can
// name are a cell's center and a cell's outer edge. Which one an end is decides
// what a line meeting it there becomes: a line ending at a center and another
// leaving that center form a corner, while a line running out through an edge
// carries on past whatever meets it, which is a tee.
//
// This is the distinction a vector graphics stroke draws between a butt cap and
// a square cap, where the square cap extends the stroke by half its width past
// the endpoint. Half a stroke here is half a cell.
//
// Unicode has a glyph for a line reaching only the middle of its cell (U+2574
// through U+2577), so a `Center` end is drawn as one and the reader sees where
// the line really stops rather than having to infer it from the junctions. With
// `Charset::Ascii` there is nothing to draw half a line with, so both ends fill
// their cell and only the junctions around them say which was which.
enum class LineEnd : int8_t {
  // The line stops at the center of its end cell. Lines meeting there corner.
  Center,
  // The line runs out through the outer edge of its end cell, joining whatever
  // is beyond it. Lines meeting there tee.
  Edge,
};

// A grid of styled cells staged for rendering to a terminal.
//
// Coordinates are 0-based with (0, 0) at the top left, `x` counting terminal
// columns and `y` counting rows.
//
// A buffer renders once, top to bottom, the way a compiler writes diagnostics.
// There is no cursor addressing and nothing is ever redrawn, so a rendered
// buffer is just as valid in a file or a pipe as on a terminal.
//
// Every row is a line, ended by a newline of its own, so nothing is left for
// the terminal to break. A break introduced to fit a width is an ordinary
// newline like any other, which is what lets wrapped text carry an indent or
// sit in a column beside a gutter: a terminal wrapping a row of its own accord
// continues at column zero, under the gutter rather than beside it. It also
// means text copied out of the output holds the lines that were displayed.
//
// The cost is that such a break is in whatever a reader copies, so wrapping
// never puts one inside a word. A path or a URL stays whole and overhangs the
// width when it doesn't fit, which is what keeps it selectable in one piece and
// clickable where a terminal recognizes one. Wrapping only adds breaks as well:
// the newlines already in a caller's text are kept as they are. A row is a row
// once something is drawn into it, so a break the text ends with closes its
// last line rather than opening an empty one after it.
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
// A buffer is `columns()` wide, and that width is the whole point of it: it is
// what wrapping fits text into, and it comes from the terminal where one was
// measured and from `DefaultColumns` where none was. Rows are the direction
// there is no bound in -- a buffer grows downward to whatever is drawn into it,
// up to `MaxRows` -- so laying out is a question of how many rows something
// takes, never of how wide the grid will turn out to be.
//
// Coordinates are the caller's to get right. Drawing a line outside the width,
// or starting text outside it, is a programming error and is checked: a caller
// deciding where to put something already knows the width, since it is what
// decided the layout, and a drawing that lands outside it is a bug in that
// layout rather than something to silently clip. Origins are checked against
// `MaxRows` the same way, though text that runs off the bottom on its own
// newlines is clipped rather than checked, as an overhang is.
//
// A row can still end up wider than `columns()`. Text that starts inside the
// width may run off the right of it: a quoted source line longer than the room
// left, a double-width character in the last column, and above all a word
// wrapping cannot break, which is moved to a row of its own and then overhangs
// it. Breaking that word is the alternative, and it costs a reader the ability
// to copy or click it. So `width()` can exceed `columns()`, while nothing is
// ever drawn left of the origin or beyond `MaxColumns`.
//
// A combining mark renders into the cell before it, so one with no cell before
// it -- at column zero, or on a row nothing has been drawn on -- has nowhere to
// go and is dropped. That is data rather than a coordinate, which is why it is
// dropped rather than checked: source files contain such text.
//
// TODO: None of this handles bidirectional text. A right-to-left run reorders
// on screen, so the column a character occupies stops following from the
// characters before it, which is the assumption every position here rests on:
// that drawing advances left to right by the width of what was drawn. Getting
// this right needs the reordering to happen before anything is placed, which
// makes it a question about where the boundary between a client's layout and
// this buffer should sit -- whether the buffer takes runs that are already in
// visual order, or takes logical order and reorders as it draws, and what it
// then means for a caller to name a column at all. Marking a span and drawing a
// line under it are the hard cases, since a logically contiguous span need not
// be contiguous on screen.
class Buffer {
 public:
  // The bounds a buffer exists within.
  //
  // These are far past anything a terminal displays, and exist so that a cell
  // index stays representable rather than to ration anything. `columns()` and
  // every row drawn into are checked against them, so a caller cannot reach
  // outside them by asking. What can reach `MaxColumns` without being asked for
  // is a word overhanging the target width, and that alone is clipped rather
  // than checked, since how far it overhangs is a fact about the text.
  static constexpr int MaxColumns = 1 << 14;
  static constexpr int MaxRows = 1 << 16;

  // The most bytes of text one operation draws or measures.
  //
  // The column advances by the width of what was drawn whether or not a cell
  // was written, so without this a long enough run would carry it past what an
  // `int` holds and come back negative. Far more text than any terminal shows,
  // and a caller with this much has built it rather than read it off a line.
  static constexpr int MaxTextBytes = 1 << 24;

  // The widest tab stops a buffer draws to.
  //
  // Far past any terminal, and small enough that even text made entirely of
  // tabs measures into a column an `int` holds: a tab is the one character
  // that occupies more columns than it does bytes, so this is what bounds
  // `MaxTextBytes` of them.
  static constexpr int MaxTabWidth = 64;

  // Where a drawing ended: for text, the row it ended on and the column after
  // its last code point there; for a line or a box, the cell past the end of
  // what it drew.
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

  // Constructs an empty buffer holding `charset`, laying out for
  // `DefaultColumns`.
  explicit Buffer(Charset charset) : Buffer(DefaultColumns, charset) {}

  // Constructs an empty buffer `columns` wide, which must be in
  // [1, `MaxColumns`], and whose tabs advance to stops `tab_width` columns
  // apart.
  //
  // The width is what everything drawn into the buffer is laid out for and
  // checked against, not a starting size. The grid holds it from the start, so
  // a row is only ever reallocated for something that overhangs it.
  Buffer(int columns, Charset charset, int tab_width = DefaultTabWidth);

  // Constructs an empty buffer holding `capabilities`'s charset and tab stops,
  // laying out for its width, or for `DefaultColumns` where it has none.
  //
  // Both numbers are clamped rather than checked. They describe a terminal
  // rather than coming from a caller -- `columns` by way of `COLUMNS`, which
  // anyone can export as anything -- so a value a grid cannot hold is bad input
  // rather than a mistake, and the nearest usable one lays out no worse than
  // the fallback would.
  explicit Buffer(const Capabilities& capabilities)
      : Buffer(std::clamp(capabilities.columns.value_or(DefaultColumns), 1,
                          MaxColumns),
               capabilities.charset,
               std::clamp(capabilities.tab_width, 1, MaxTabWidth)) {}

  // Returns the width everything drawn into the buffer is laid out for.
  auto columns() const -> int { return columns_; }

  // Returns the columns the grid currently holds: `columns()` until something
  // overhangs it, and at least enough to hold the overhang after that.
  auto width() const -> int { return width_; }

  // Returns the number of rows the grid holds, which is one past the last row
  // drawn into.
  auto height() const -> int;

  auto charset() const -> Charset { return metrics_.charset(); }

  // Returns how text is measured for this buffer's charset.
  //
  // The buffer lays its cells out with this, so a caller deciding where to put
  // something asks the same thing the drawing will.
  auto metrics() const -> Metrics { return metrics_; }

  // Returns where `DrawText` would end for these arguments, without drawing.
  //
  // Measuring and drawing walk the text with the same code, differing only in
  // whether they write a cell, so a layout decision made from this can't
  // disagree with what drawing then does.
  //
  // This is for text that a tab, a newline, or a carriage return makes
  // positional. Text with none of them is as wide wherever it is drawn, and
  // `Metrics::Width` answers for it without a buffer to draw into.
  auto MeasureText(int x, int y, int margin, llvm::StringRef text) const
      -> DrawEnd;

  // Returns where the `DrawText` taking no margin would end, which draws `text`
  // as text of its own beginning at (x, y).
  auto MeasureText(int x, int y, llvm::StringRef text) const -> DrawEnd {
    return MeasureText(x, y, x, text);
  }

  // Returns where `DrawWrappedText` would end for these arguments, without
  // drawing.
  //
  // The block and the origin are checked as drawing checks them, so measuring
  // answers only for arguments drawing would accept.
  auto MeasureWrappedText(int x, int y, int margin, int max_width,
                          llvm::StringRef text) const -> DrawEnd;

  // Returns the fewest columns `text` wraps into without overhanging them,
  // which is the width of its widest word since wrapping never breaks one.
  //
  // Wrapping into fewer columns still draws everything; the excess overhangs.
  // So this is a layout preference rather than a minimum.
  auto MeasureWrapWidth(llvm::StringRef text) const -> int;

  // Draws `code_point` at (x, y), which must be inside `columns()` and
  // `MaxRows`, adding rows as needed to reach it.
  //
  // Returns the column after it, which is `x` again for a combining mark since
  // one renders into the column before it. A double-width character starting in
  // the last column is drawn rather than refused, and takes the column after
  // it: half a character is not something a terminal can render, so the choice
  // is between the whole of it and none, and this is the same overhang wrapping
  // allows a word that fits no row.
  auto DrawCodePoint(int x, int y, char32_t code_point, const Style& style)
      -> DrawEnd;

  // Draws a horizontal line across `length` columns starting at (x, y).
  //
  // By default the line runs between the centers of its first and last cells,
  // which is what a line connecting two things is: `DrawBox` draws its four
  // sides this way, and each pair meets at a corner. `LineEnd::Edge` instead
  // runs that end out through the side of its cell, which is what a line
  // bounding `length` whole columns of something is, and what makes a line
  // meeting it there a tee. A line of one column between two centers is a
  // point, and is drawn as one.
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
  auto DrawHorizontalLine(int x, int y, int length, const Style& style,
                          LineEnd start = LineEnd::Center,
                          LineEnd end = LineEnd::Center) -> DrawEnd;

  // Draws a vertical line down `length` rows starting at (x, y), with the same
  // meaning for its ends. Returns the row after it, in the column it ran down.
  auto DrawVerticalLine(int x, int y, int length, const Style& style,
                        LineEnd start = LineEnd::Center,
                        LineEnd end = LineEnd::Center) -> DrawEnd;

  // Draws the outline of a box with its top-left corner at (x, y).
  //
  // Each side runs between the centers of the cells it ends in, so the four
  // corners come out of the sides meeting there. A box with no interior is
  // then the single line that bounds it, and one with no extent in either
  // direction is a point, without either being a case of its own.
  auto DrawBox(int x, int y, int box_width, int box_height, const Style& style)
      -> DrawEnd;

  // Draws `text` starting at (x, y), which must be inside `columns()`, as part
  // of text whose left edge is `margin`.
  //
  // Nothing here wraps, so text with no newline in it runs off the right of the
  // width when it is longer than the room left, exactly as an overhanging word
  // does. That is what this is for: a source line is quoted as it was written,
  // and deciding how much of one to show is the caller's, made against
  // `columns()` before the quoting starts.
  //
  // Newlines return to column `margin` on the next row, carriage returns to
  // column `margin` on the same row, and tabs advance to the next tab stop,
  // with stops measured from `margin` so that a quoted source line keeps the
  // tab alignment it had in the file wherever the quote is placed. Returns
  // where it ended, which for text with a newline in it is on a later row than
  // it started.
  //
  // The margin is what lets text with newlines in it be drawn as differently
  // styled spans, each starting where the last ended and all naming the same
  // margin, the way `DrawWrappedText` does for a block: a newline in the middle
  // of such a run returns to the text's own left edge rather than to wherever
  // the span it fell in happened to start.
  auto DrawText(int x, int y, int margin, llvm::StringRef text,
                const Style& style) -> DrawEnd;

  // Draws `text` as text of its own beginning at (x, y), which is then both
  // where it starts and the margin its later rows return to.
  auto DrawText(int x, int y, llvm::StringRef text, const Style& style)
      -> DrawEnd {
    return DrawText(x, y, x, text, style);
  }

  // Draws `text` starting at (x, y), into the block of `max_width` columns
  // beginning at `margin`.
  //
  // The block must lie within `columns()` and `x` within the block, so
  // `0 <= margin <= x < margin + max_width <= columns()`. A block is a division
  // of the width rather than something that can exceed it: what a caller wants
  // when it has nothing to divide is `max_width` of `columns() - margin`, the
  // whole of what is left.
  //
  // The block is what the text wraps within, and (x, y) is only where this run
  // of it starts: rows after the first begin at `margin`, and how much room a
  // row has is measured from there. A block whose spans are styled differently
  // is drawn as one call per span, each starting where the last ended and all
  // naming the same margin and width. Passing `x` as the margin draws a block
  // in one call.
  //
  // Wrapping breaks at ASCII spaces, tabs, and carriage returns, and only
  // there. A word here is whatever lies between two of them, so a URL is one
  // word, and one too long for a row of its own is moved down to one and then
  // overhangs it rather than being broken.
  //
  // Whitespace stops at the block's edge rather than running past it, so the
  // spaces between two words stay on the row the first of them ended and the
  // row the second wraps onto begins at the margin. Spaces the text opens with,
  // or that follow a newline in it, are kept as they are, since those are
  // indentation the caller wrote.
  //
  // Newlines are breaks the caller already made, and are kept as they are:
  // wrapping only adds breaks to the text it is given. They break the line as a
  // wrap does, continuing at `margin` on the next row, and carriage returns are
  // dropped so that CRLF endings break exactly once.
  //
  // A tab is both a break opportunity and a jump to the next tab stop, with
  // stops measured from `margin` rather than from `x`. The margin is the one
  // column every row of the block begins at, so the stops are the same on each
  // of them and a tabbed column stays a column however the text wraps; stops
  // from `x` would move with the span that happened to be drawn first. A tab
  // that would reach past the block stops at its edge, like the spaces do,
  // leaving the word after it to wrap.
  //
  // `DrawText` is the way to draw text that should not wrap at all, and differs
  // in more than that: it keeps every space, and returns to the margin on a
  // carriage return rather than dropping it.
  //
  // Returns where it ended.
  //
  // TODO: There is no mode that reflows, treating the newlines in `text` as
  // breaks to be chosen again rather than kept. Text that arrives wrapped to
  // some other width keeps that wrapping, which is wrong for it wherever that
  // width isn't the one it is being drawn into. Add one when there is a caller
  // with such text, since which breaks a reflow may discard -- every newline,
  // or only those a previous wrapping introduced -- is a question about where
  // that text came from.
  auto DrawWrappedText(int x, int y, int margin, int max_width,
                       llvm::StringRef text, const Style& style) -> DrawEnd;

  // Renders the grid, appending the bytes that draw it to `out`.
  //
  // Each row ends in a newline, with trailing blank cells dropped so output
  // carries no invisible padding. The rendering ends with the style turned off
  // so nothing bleeds into what is printed next, and a style that paints blank
  // cells is turned off at each row's end so a background does not run to the
  // right edge. Color is chosen here rather than at construction because it
  // affects only how cells are serialized, while the charset decides how
  // content is laid out into them.
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
  // The directions in which drawn lines leave a cell, and whether the cell
  // holds line art at all. A cell's glyph is a function of the directions
  // alone.
  enum LineDirection : uint8_t {
    LineLeft = 1 << 0,
    LineRight = 1 << 1,
    LineUp = 1 << 2,
    LineDown = 1 << 3,
    LineDirections = 0b1111,
    // Set on every cell line drawing writes. A cell can hold line art and no
    // directions -- a line between one center and itself is a point -- and
    // without this such a cell would be indistinguishable from one holding
    // text, so nothing drawn later would join it.
    LineCell = 1 << 4,
  };

  struct Cell {
    // The code point rendered here. For a cell with `lines` set, this is
    // derived from those bits and the charset.
    char32_t code_point = ' ';

    Style style;

    // Which directions drawn lines leave this cell in, with `LineCell` set,
    // or zero for a cell holding text.
    uint8_t lines = 0;

    // Whether this cell is the right half of a double-width character, and so
    // renders nothing of its own.
    bool is_continuation = false;
  };

  // Checks that `text` is short enough to measure without overflowing a column.
  static auto CheckTextSize(llvm::StringRef text) -> void {
    CARBON_CHECK(text.size() <= MaxTextBytes,
                 "Laying out {0} bytes of text is past the {1} one operation "
                 "handles.",
                 text.size(), MaxTextBytes);
  }

  auto CellIndex(int x, int y) const -> int { return y * width_ + x; }
  auto CellAt(int x, int y) -> Cell& { return cells_[CellIndex(x, y)]; }
  auto CellAt(int x, int y) const -> const Cell& {
    return cells_[CellIndex(x, y)];
  }

  // Checks that (x, y) is somewhere a drawing may start.
  //
  // The text walks check this themselves, together with the bounds particular
  // to each: they are inlined into every text operation, and one check there
  // costs measurably less than two.
  auto CheckOrigin(int x, int y) const -> void {
    CARBON_CHECK(
        x >= 0 && x < columns_ && y >= 0 && y < MaxRows,
        "Drawing at ({0}, {1}) is outside the {2} columns and {3} rows "
        "a buffer covers.",
        x, y, columns_, MaxRows);
  }

  // Places `code_point` at (x, y) without checking it against the target width
  // or `MaxRows`, which text reaches on its own by overhanging or by carrying
  // newlines. Past either, nothing is drawn and the column still advances. The
  // coordinates must be non-negative, which follows from the origin the walk
  // was checked at.
  auto PlaceCodePoint(int x, int y, char32_t code_point, const Style& style)
      -> int;

  // The walks behind the text operations, over which drawing and measuring are
  // the same code. `place` is called with each code point and where it goes,
  // and returns the column after it: `PlaceCodePoint` when drawing, and the
  // width alone when measuring.
  template <typename PlaceFn>
  auto WalkText(int x, int y, int margin, llvm::StringRef text,
                PlaceFn place) const -> DrawEnd;
  template <typename PlaceFn>
  auto WalkWrappedText(int x, int y, int margin, int max_width,
                       llvm::StringRef text, PlaceFn place) const -> DrawEnd;

  // Adds rows until row `y` exists.
  auto EnsureRow(int y) -> void;

  // Widens the grid until column `x` exists, reflowing the rows it already
  // holds, which are stored back to back. Only something overhanging the target
  // width reaches past it, so this runs for nothing else.
  auto EnsureColumn(int x) -> void;

  // Resets the cells in row `y` spanning columns [x, x + width), along with
  // either half of a double-width character that straddles the range's edges.
  auto ClearCells(int x, int y, int width) -> void;

  // Appends `code_point` to the marks rendered with the cell before column `x`.
  auto AttachCombiningMark(int x, int y, char32_t code_point) -> void;

  // Adds `directions` to the lines through (x, y) and updates its glyph.
  auto DrawLine(int x, int y, uint8_t directions, const Style& style) -> void;

  // Returns the last column in row `y` that renders anything under `mode`, or
  // -1 when the row renders nothing.
  auto LastVisibleColumn(int y, ColorMode mode) const -> int;

  // The width laid out for, and the width the grid holds. They differ only
  // where something overhung the first.
  int columns_;
  int width_;

  int tab_width_;
  Metrics metrics_;

  llvm::SmallVector<Cell, 0> cells_;

  // Combining marks, as UTF-8, for the few cells that have any, keyed by cell
  // index. Kept out of `Cell` so that the common case of no marks costs
  // nothing per cell. Always empty under `Charset::Ascii`.
  llvm::DenseMap<int, std::string> combining_marks_;
};

}  // namespace Carbon::Terminal

#endif  // CARBON_COMMON_TERMINAL_BUFFER_H_

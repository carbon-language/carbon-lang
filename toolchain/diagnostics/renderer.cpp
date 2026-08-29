// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/renderer.h"

#include <algorithm>
#include <string>
#include <utility>

#include "common/check.h"
#include "common/terminal/buffer.h"
#include "common/terminal/color.h"
#include "common/terminal/metrics.h"
#include "common/terminal/style.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace Carbon::Diagnostics {

using Terminal::AnsiColor;
using Terminal::Buffer;
using Terminal::Charset;
using Terminal::LineEnd;
using Terminal::Metrics;
using Terminal::Style;

// The columns from the frame to the content beside it: the frame itself, and
// the column next to it that a source row leaves blank.
static constexpr int FrameToContentWidth = 2;

// Points, from the margin outside the line numbers, to the one row to begin
// reading at: the source row the leading part's text is placed under, or that
// part's location row when it has no source. A merged snippet can start on
// another line, so this shows which row the problem is on.
//
// Two ASCII characters rather than one of the arrows, which all read better and
// none of which measure reliably: they are East Asian Ambiguous, so a font may
// draw one across two columns while the width tables call it one, and a
// terminal falling back to such a font pushes the whole row over or clips the
// glyph. This is two columns because it is two characters, so nothing can
// disagree about it, and a font with programming ligatures draws it as an arrow
// anyway.
static constexpr llvm::StringLiteral LineArrow = "->";

// The columns the arrow spends before the line numbers. No space after it: the
// numbers are right-aligned in a field of their own, so a number shorter than
// that field already leaves a gap, and a number filling it is the widest there
// is to keep the arrow off.
static constexpr int MarginWidth = LineArrow.size();

// The columns a location row spends before the location it names: `╭─┤ `.
static constexpr int LocationPrefixWidth = 4;

// The columns a label line or a path row's line spends reaching its text: three
// of line, and then either a space or, where it reaches a label, the bar
// framing it.
static constexpr int LineToTextWidth = 4;

// The columns the bar framing a label's left side spends before its text:
// `┤ `. The bar runs down every row the label takes, so that what belongs to
// the label is bounded rather than left to be inferred from indentation.
static constexpr int LabelFrameWidth = 2;

// The columns between a label line and its label's text: the label line
// reaching the bar that frames the text, and the bar's own space before the
// words.
static constexpr int LabelIndentWidth = LineToTextWidth - 1 + LabelFrameWidth;

// The columns each label's text starts right of where the text above it
// starts, so the indentation steps with the order the rows are read in.
static constexpr int LabelStepWidth = 2;

// The columns of the close, `╰────`, starting at the frame. The blank line
// after a diagnostic separates it from the next, so the close only needs to
// turn the frame's corner.
static constexpr int CloseRuleWidth = 5;

// The columns of source a diagnostic aims to show: the width code is formatted
// to, so a formatted line is shown whole and nothing wider is asked for than
// that needs. A line longer than this is windowed, the way one longer than the
// terminal is.
static constexpr int FormattedSourceColumns = 80;

// The columns a line number is given in the gutter, so that the frame runs
// down one column for every diagnostic in a file under ten thousand lines
// rather than stepping with each number's digits, at a cost of at most three
// columns on the smallest files. A number needing more still fits -- the frame
// moves right and the source is squeezed by the difference -- so the width
// laid out against is the same whatever line a diagnostic lands on.
static constexpr int LineNumberColumns = 4;

// The most bytes one drawn string may hold. Nothing legitimate comes near it:
// it exists so that pathological input -- a generated file with one enormous
// line, or a message argument holding one -- degrades by truncation instead of
// hitting a limit the terminal buffer CHECKs. It bounds both the bytes of a
// single draw and, through the rows a wrapped string of this many bytes takes,
// the height a handful of them can reach, so it stays well under the buffer's
// row limit.
static constexpr size_t MaxDrawnBytes = 1 << 16;

// Columns between tab stops, matching what a terminal does with a tab.
static constexpr int SourceTabWidth = 8;

// What is drawn at each end of a source line that windowing elided.
static constexpr llvm::StringLiteral WindowEllipsis = "...";

// The fewest columns of source a window shows, however narrow the terminal
// is.
static constexpr int MinWindowColumns = 16;

// Diagnostics quote code between backticks, and a quoted run is one word to
// wrapping, so a type written with spaces in it, or a snippet of syntax, is
// never split across rows.
static constexpr bool BackticksAreQuotes = true;

// The most lines shown rather than elided between two spans in one file. One
// line of context is worth as much as the row saying it was skipped.
static constexpr int MaxShownGap = 1;

// What a location row names for a part with no location at all, whose text is
// still placed under a location row.
static constexpr llvm::StringLiteral NoLocationText = "<no location>";

// The fewest columns a framed diagnostic is drawn in, below which the compact
// form is used instead. A frame, its gutter, and a label hanging off a span
// need room before any of them say anything, and past some point there isn't
// enough to be worth spending on structure. Low enough that a terminal narrow
// enough to hit it is one a label would route on anyway.
//
// Where that point is, is a guess. It is not derived from anything, and no
// layout depends on the number being right -- a terminal one column either side
// of it gets a rendering that works. Move it if it turns out to be wrong.
static constexpr int MinFrameColumns = 60;

// Returns the word naming `level`, including its separator.
static auto LevelText(Level level) -> llvm::StringRef {
  switch (level) {
    case Level::Error:
      return "error: ";
    case Level::Warning:
      return "warning: ";
  }
}

// Returns the style the underlines under the source take, which is the color of
// whatever they belong to without its weight. Bold reaches the line-drawing
// glyphs unevenly -- several terminals leave them alone and others only
// brighten them -- so the weight says nothing reliable there while it still
// says something on words.
//
// `RotateUnderlineStyles` makes one exception: the heavy underline keeps its
// bold. A terminal that ignores bold on line art loses nothing there, and one
// that renders it matches the underline to the bold text placed under it.
static auto UnderlineStyle(const Style& style) -> Style {
  return style.Bold(false);
}

// The colors one theme is drawn in: what a diagnostic of that level, or the
// notes explaining one, take for their words and for the underlines under the
// source that belong to them.
//
// The second is the theme's own color, which a range alone on its line takes.
// Where several share a line, the whole ramp is handed out left to right
// starting at the dark end, so a color always says which theme a range belongs
// to and only then which range it is.
using Theme = llvm::ArrayRef<Terminal::Color>;

// The themes for a terminal that renders 24-bit color, where the colors are
// ours to choose and are chosen together.
//
// Each theme is a ramp handed out left to right across the ranges on a line:
// darker, the theme's own color, lighter. The three centers sit at one
// lightness, so the level a diagnostic is reported at is said by hue and by the
// word rather than by how loud it is, and lightness varies only inside a theme,
// where a step says "a different range" rather than "a more important one".
//
// There are two palettes because a color picked to read against black is hard
// to read against white. Which is used follows `Capabilities::background`.
//
// See /toolchain/docs/diagnostics_rendering.md#the-palette for where these
// values came from and the distances and contrast ratios they satisfy. Changing
// one means checking it against those, not against how it looks on a swatch:
// these are drawn as underlines a pixel tall.
static constexpr Terminal::Color DarkBgErrorTheme[] = {
    {0xb5, 0x56, 0x00}, {0xff, 0x72, 0x61}, {0xff, 0xbc, 0xc8}};
static constexpr Terminal::Color DarkBgWarningTheme[] = {
    {0x51, 0x86, 0x00}, {0xac, 0xac, 0x00}, {0xff, 0xc9, 0x21}};
static constexpr Terminal::Color DarkBgNoteTheme[] = {
    {0x00, 0x82, 0x9a}, {0x00, 0xbc, 0xbc}, {0x00, 0xf3, 0xc6}};

static constexpr Terminal::Color LightBgErrorTheme[] = {
    {0x80, 0x35, 0x00}, {0xd0, 0x00, 0x00}, {0xff, 0x28, 0x6a}};
static constexpr Terminal::Color LightBgWarningTheme[] = {
    {0x3d, 0x59, 0x00}, {0x73, 0x73, 0x00}, {0xad, 0x8c, 0x00}};
static constexpr Terminal::Color LightBgNoteTheme[] = {
    {0x00, 0x59, 0x67}, {0x00, 0x7f, 0x7f}, {0x00, 0xa7, 0x8e}};

// The themes for a terminal with only the 16 named colors, which renders them
// through the user's own palette. Their appearance is not ours to choose, so
// these say only which entry to use and let the user's theme decide the rest --
// including which background it goes with, which is why these don't vary with
// it.
//
// Each is a ramp of two rather than three, the plain entry and then the bright
// one, because there is nowhere for a longer one to go. Downsampling the 24-bit
// colors instead would be worse than stopping at two: two of them land on
// `BrightBlack` and on `White`, which are not that theme or any other. Which
// theme a range belongs to is worth more than telling two ranges of one theme
// apart, so a ramp runs out and repeats rather than reaching outside its theme.
static constexpr Terminal::Color NamedErrorTheme[] = {AnsiColor::Red,
                                                      AnsiColor::BrightRed};
static constexpr Terminal::Color NamedWarningTheme[] = {
    AnsiColor::Yellow, AnsiColor::BrightYellow};
static constexpr Terminal::Color NamedNoteTheme[] = {AnsiColor::Cyan,
                                                     AnsiColor::BrightCyan};

// Returns whether `capabilities` renders colors chosen here rather than the
// user's.
//
// `Ansi256` is on this side of the line because rounding to its cube keeps
// every theme's three colors distinct and keeps two colors of different themes
// far enough apart to read as different themes -- comfortably so on a dark
// background, and closer on a light one, where an error and a warning range
// sharing a line is the arrangement to watch. `Ansi16` keeps neither, which is
// why it has a table of its own -- and why the background makes no difference
// there, since a named color is drawn from a palette the user chose to go with
// their background.
static auto HasChosenColors(const Terminal::Capabilities& capabilities)
    -> bool {
  return capabilities.color_mode == Terminal::ColorMode::Ansi256 ||
         capabilities.color_mode == Terminal::ColorMode::Truecolor;
}

// Returns whether the chosen colors should be the ones for a light background.
static auto WantsLightBackground(const Terminal::Capabilities& capabilities)
    -> bool {
  return capabilities.background == Terminal::Background::Light;
}

// Returns the theme a diagnostic at `level` is drawn in.
static auto LevelTheme(Level level, const Terminal::Capabilities& capabilities)
    -> Theme {
  bool light = WantsLightBackground(capabilities);
  if (!HasChosenColors(capabilities)) {
    return level == Level::Error ? Theme(NamedErrorTheme)
                                 : Theme(NamedWarningTheme);
  }
  switch (level) {
    case Level::Error:
      return light ? Theme(LightBgErrorTheme) : Theme(DarkBgErrorTheme);
    case Level::Warning:
      return light ? Theme(LightBgWarningTheme) : Theme(DarkBgWarningTheme);
  }
}

// Returns the theme anything explaining a diagnostic is drawn in.
static auto ExplanationTheme(const Terminal::Capabilities& capabilities)
    -> Theme {
  if (!HasChosenColors(capabilities)) {
    return Theme(NamedNoteTheme);
  }
  return WantsLightBackground(capabilities) ? Theme(LightBgNoteTheme)
                                            : Theme(DarkBgNoteTheme);
}

// The place in a theme's ramp that is the theme's own color, which its words
// are drawn in and which a range takes when it is the only one of its theme on
// a line. Every ramp puts it second, so a ramp of two runs up to it and a ramp
// of three runs through it.
static constexpr int ThemeCenter = 1;

// Returns the style a theme's words are drawn in. Also the color of the
// underline beneath the source that belongs to it, so that the underline and
// the sentence explaining it look like one thing.
static auto ThemeStyle(Theme theme) -> Style {
  return Style().Bold().Foreground(theme[ThemeCenter]);
}

// The word leading anything that explains the message rather than stating it.
static constexpr llvm::StringRef ExplanationText = "note: ";

// The style of the frame and the line numbers beside it. Not dim, which several
// terminals ignore, leaving the frame as prominent as the code inside it.
static auto FrameStyle() -> Style {
  return Style().Bold().Foreground(AnsiColor::BrightBlue);
}

// The style of the diagnostic kind, which is only present under a flag and is
// never the thing being read.
static auto KindStyle() -> Style { return Style().Dim(); }

// The style of location information, which says how the code below it was
// reached rather than anything about the code itself. Dim so that a reader who
// already knows the path can skip it, and unlike the frame nothing is lost on a
// terminal that ignores dim. It covers the whole row, quoted code included.
static auto PathStyle() -> Style { return Style().Dim(); }

namespace {

// The glyphs that vary with the character set.
//
// Everything else the frame is made of comes from `Terminal::Buffer`'s line
// drawing, which picks its own glyphs and forms the corners and tees where
// lines meet.
struct Glyphs {
  // The frame's glyph on underline rows and label rows, so that which rows are
  // the reader's code is visible at a glance.
  char32_t dotted_frame;
  // Drawn where lines of source were skipped.
  char32_t elision;

  static auto For(Charset charset) -> Glyphs {
    if (charset == Charset::Utf8) {
      return {.dotted_frame = U'·', .elision = U'┆'};
    }
    // ASCII has one broken bar for both the underline rows and the elision. A
    // row holding nothing but the bar is an elision; one with an underline
    // after it is not.
    return {.dotted_frame = U':', .elision = U':'};
  }
};

}  // namespace

// Returns whether a terminal decoding `charset` draws `byte` as itself.
static auto IsPrintable(Charset charset, char byte) -> bool {
  auto value = static_cast<unsigned char>(byte);
  if (charset == Charset::Ascii) {
    // A terminal decoding some single-byte encoding renders anything outside
    // printable ASCII as something, and there is no telling what.
    return value >= 0x20 && value < 0x7f;
  }
  // Everything else goes to the buffer, which decodes UTF-8 and replaces what
  // has no rendering of its own.
  return value >= 0x20 && value != 0x7f;
}

namespace {

// A count of bytes into a source line, as against a column on screen.
//
// A `Loc` gives its span as byte offsets -- `column_number` counts bytes
// despite its name, and Clang's land inside multi-byte characters -- while
// everything drawn is placed by column. The two count the same line and
// neither follows from the other, so this keeps them from being taken for each
// other in the one place both are in scope. Columns stay `int`, which is what
// `Terminal::Buffer` addresses its grid with, so the count that is not a
// coordinate is the one given this type.
//
// TODO: Move this out of this file, and use it for the byte counts elsewhere
// that are called columns, starting with `Loc::column_number`.
class Bytes {
 public:
  Bytes() = default;
  explicit constexpr Bytes(int count) : count_(count) {}

  constexpr auto count() const -> int { return count_; }

  constexpr auto operator+(Bytes rhs) const -> Bytes {
    return Bytes(count_ + rhs.count_);
  }
  constexpr auto operator-(Bytes rhs) const -> Bytes {
    return Bytes(count_ - rhs.count_);
  }
  constexpr auto operator++() -> Bytes& {
    ++count_;
    return *this;
  }
  constexpr auto operator--() -> Bytes& {
    --count_;
    return *this;
  }

  friend auto operator<=>(Bytes, Bytes) = default;
  friend auto operator==(Bytes, Bytes) -> bool = default;

 private:
  int count_ = 0;
};

}  // namespace

namespace Internal {

// A source line prepared for drawing, and where a part's span lands in it.
//
// The text has tabs expanded and bytes with no rendering escaped, so that the
// columns measured against it are the columns a terminal will use.
struct PreparedSource {
  std::string text;
  int span_start = 0;
  int span_length = 1;

  // Returns the columns the source and its underline occupy together. The
  // underline can reach one column further, when a span starts past the end of
  // the line it names.
  auto Width(Metrics metrics) const -> int {
    return std::max(metrics.Width(text), span_start + span_length);
  }

  // The column the label line runs down, which layout slides between where the
  // span starts and its middle. Until it is set it is the middle, which is what
  // a span with no label of its own is measured from.
  int line_column = 0;

  // The column the label's text starts in. Layout sets it right of every
  // label line on the line, and `LabelStepWidth` right of where the text of the
  // nearest label above that is not out-dented starts.
  int label_x = 0;

  // Where the underline stops at each end. Layout sets an end to `Center` when
  // another range begins or ends in the next column, so that the two underlines
  // don't run together; an end with nothing beside it covers its whole column.
  LineEnd left_end = LineEnd::Edge;
  LineEnd right_end = LineEnd::Edge;

  // Whether the label line is drawn in place of the underline rather than
  // hanging off it. An underline of one column with an end pulled in has no
  // junction to offer, and the line down its column says the column is
  // underlined without saying the underline carries on anywhere.
  bool label_line_stands_in = false;

  // Returns the column in the middle of the span.
  auto Middle() const -> int { return span_start + span_length / 2; }

  // Returns whether the underline covers `column` at all.
  auto Covers(int column) const -> bool {
    return column >= span_start && column < span_start + span_length;
  }

  // Returns whether the underline runs out through the left or the right side
  // of `column`. An end pulled in to clear the range beside it gives up the
  // side it was pulled in on, which is the side a label line could not tee
  // into.
  auto ReachesLeftOf(int column) const -> bool {
    return Covers(column) && (column > span_start || left_end == LineEnd::Edge);
  }
  auto ReachesRightOf(int column) const -> bool {
    return Covers(column) && (column < span_start + span_length - 1 ||
                              right_end == LineEnd::Edge);
  }

  // Returns whether any column of the underline runs out through both of its
  // sides, which is what a label line needs one to do to tee into it.
  auto Joinable() const -> bool {
    if (span_length == 1) {
      return left_end == LineEnd::Edge && right_end == LineEnd::Edge;
    }
    return span_length > 2 || left_end == LineEnd::Edge ||
           right_end == LineEnd::Edge;
  }
};

// What a part of a diagnostic is to the reader, which is what decides how it
// is drawn.
enum class Part : int8_t {
  // A step in the path by which the part after it was reached.
  LocationInfo,
  // Something read against the code to explain the problem rather than to
  // state it: a label, or the message itself once a context is stating the
  // problem in its place.
  Explanation,
  // The problem itself: whatever leads the diagnostic, and the source that is
  // directly part of it.
  Problem,
};

// A part of a diagnostic prepared for drawing.
struct PreparedPart {
  // What this is to the reader.
  Part role = Part::Explanation;

  // Whether this part states the problem: the message, or the context that
  // `LeadingContext` returns. Its text begins with the level word and is
  // bold, and its underline is the heavy one.
  //
  // TODO: The design has a context scope attach a label instead of leading the
  // diagnostic. Make the message always lead once contexts are labels.
  bool is_leading = false;

  // Whether this part's range is drawn as the heavy underline: the range the
  // leading part's text is placed under, or a problem range identical to it.
  // It keeps the level's center color and is left out of the ramp. Set
  // during layout.
  bool is_heavy_underline = false;

  // Whether this part's range came from the message's own location rather than
  // from a range attached to the diagnostic. Only the message has one.
  //
  // An attached problem range containing the location takes its place: the
  // message's text is placed under that range, and the location has no
  // underline of its own. See `FindMessageHost`.
  bool from_message_loc = false;

  // The word leading this part's text, including its separator, and empty for
  // location information, which leads with its own words instead.
  llvm::StringRef level_text;

  // The style of that word and of this part's text.
  Style style;

  // The style of the underline under this part's source, and of the label line
  // and bar of the label hanging off it. The rotation moves this along the
  // theme's ramp, which is why it is not `style`: the level word, the arrow in
  // the margin, and the line number beside the reported line all say what the
  // diagnostic is, so where a range sits on a line must not recolor them.
  Style underline_style;

  // `file:line:column`, with unknown parts left off, and empty for a part
  // with no location at all.
  std::string location;

  // The last byte column of the range starting at `location`, or 0 for a
  // range of one column or of unknown extent. The compact form appends it to
  // the location as `-<end>`, which is all it has to say the extent with; the
  // framed form says it with the underline instead.
  int32_t location_end_column = 0;

  // The part's text when it is placed under the part's location row, which is
  // for any part with no source. Empty when the text is a label.
  std::string text;

  // The part's text when it is placed under the underline on its source, which
  // is for any part that has source. Empty otherwise.
  std::string label;

  // ` [Kind]`, or empty when kinds aren't being included. It goes at the end
  // of the last row carrying the part's words, so that a test matching on it
  // always finds it last, and so that a part drawn on two rows doesn't get it
  // twice.
  std::string kind_suffix;

  // Returns the suffix for a row drawing this part's text. The text goes on
  // exactly one kind of row -- a label row when the part has source, a location
  // label row otherwise -- so a row of the other kind gets no suffix.
  auto KindSuffixOn(bool is_label_row) const -> llvm::StringRef {
    return is_label_row == label.empty() ? llvm::StringRef() : kind_suffix;
  }

  // The source and its span. Only meaningful when `has_source` is set.
  PreparedSource source;
  bool has_source = false;

  // Where the source came from, for grouping spans that share a file and for
  // finding the lines between two of them.
  llvm::StringRef file;
  llvm::StringRef file_text;
  llvm::StringRef raw_line;
  int line_number = 0;
};

}  // namespace Internal

using Internal::Part;
using Internal::PreparedPart;
using Internal::PreparedSource;

// Returns `style` recolored for the `index`th of `count` ranges of `theme` on a
// line, keeping everything about it except which color it is.
//
// Several ranges walk the ramp from its dark end, so that reading left to right
// reads as a progression. A range with no others of its theme beside it takes
// the theme's own color instead, so that it matches the words it belongs to.
static auto RotatedStyle(const Style& style, Theme theme, int index, int count)
    -> Style {
  if (count == 1) {
    return style.Foreground(theme[ThemeCenter]);
  }
  return style.Foreground(theme[index % theme.size()]);
}

namespace {

// How a label line meets the underline it hangs from.
enum class Attachment : int8_t {
  // Tees into the underline where it hangs from. Line drawing forms the
  // junction the same way it forms the frame's own.
  Tee,
  // Stands in for the underline, in the one column it had, because the
  // underline offers no column to tee into: an underline that stops at a cell's
  // center to clear the range beside it would form a corner there, and a corner
  // reads as an underline turning rather than as one a label line leaves.
  Through,
};

// Where a label line meets its underline, and the column it runs down.
struct Placement {
  Attachment attachment;
  int line_column;
};

// A row of the rendering, laid out before anything is measured or drawn.
struct Row {
  enum class Kind : int8_t {
    // `╭─┤ location`: the location of the snippet below it, or of a part with
    // no source whose text is placed under it.
    Location,
    // A line of source, with its number beside the frame.
    Source,
    // The underlines under the source above, one for every span on it.
    Underline,
    // The text hanging off that underline, framed by a bar the label line
    // reaches.
    Label,
    // A label that wouldn't fit hanging off its span, out-dented to the column
    // the source starts in with its own frame reaching back to the span.
    RoutedLabel,
    // The text of a part with no source, placed under its location row the way
    // a label is placed under its underline.
    LocationLabel,
    // `┆`, continuing the frame across whatever isn't contiguous: skipped
    // lines of source, or the step to the next location row.
    Elision,
    // How the location in the location row below it was reached.
    Path,
  };

  Kind kind;
  // The part this row belongs to, or null for a row that draws only frame and
  // for a source line shown between two spans.
  const PreparedPart* part = nullptr;
  // `Underline` rows: every span underlined, in the order they are drawn:
  // widest first, so that the narrowest underline on a column is what shows,
  // with the heavy underline last so that nothing repaints it.
  llvm::SmallVector<const PreparedPart*, 0> spans;
  // `Source` rows: the part whose style the margin arrow and line number take,
  // set on the one line the leading part's text is placed under, and null on
  // every other line.
  const PreparedPart* points_here = nullptr;
  // `Source` rows: the number to draw, or -1 when there is none to show.
  int line_number = -1;
  // The row's text: a line of source, or the text of a label or location label.
  std::string text;
};

}  // namespace

// Returns whether `byte` continues a UTF-8 symbol that started before it.
static auto IsContinuation(char byte) -> bool {
  return (static_cast<unsigned char>(byte) & 0xc0) == 0x80;
}

// Returns `line` with tabs expanded and unrenderable bytes escaped, along with
// where the span of `span_bytes` bytes at `column` lands in the result.
//
// TODO: Highlight the source's syntax. `PreparedSource` holds plain text, so
// this needs it to hold styled runs.
static auto PrepareSource(Metrics metrics, llvm::StringRef line,
                          Bytes span_byte, Bytes span_bytes) -> PreparedSource {
  Charset charset = metrics.charset();
  // A file written on Windows carries the carriage return into the line, and
  // drawing it would put a `<0D>` on the end of every snippet row.
  line.consume_back("\r");
  // Windowing bounds what is shown by columns, not bytes, so a line of nothing
  // but combining marks could otherwise carry any number of bytes through it.
  // This bound also keeps the byte counts below inside `int`.
  line = line.take_front(MaxDrawnBytes);
  auto size = Bytes(line.size());

  // `span_byte` is one-based, as `Loc::column_number` is. A span can name a
  // token that isn't in the line at all, at the end of a file for example, so
  // it is clamped rather than trusted: a renderer must never be the reason a
  // compiler dies while reporting a problem.
  Bytes span_begin = std::clamp(span_byte - Bytes(1), Bytes(0), size);
  // Adding the length to the start would overflow for a length this doesn't
  // expect, so it is clamped to what is left of the line first.
  Bytes span_end =
      span_begin + std::min(std::max(span_bytes, Bytes(1)), size - span_begin);

  // An offset that names a byte inside a symbol, which Clang's do, would split
  // it and leave the run before it measured as loose bytes.
  if (charset == Charset::Utf8) {
    // A span starting where the line ends names no byte to walk back from, as a
    // token synthesized by error recovery past the end of a line does.
    while (span_begin > Bytes(0) && span_begin < size &&
           IsContinuation(line[span_begin.count()])) {
      --span_begin;
    }
    while (span_end < size && IsContinuation(line[span_end.count()])) {
      ++span_end;
    }
  }

  PreparedSource result;
  int cursor = 0;
  int start_column = 0;
  int end_column = 0;
  Bytes run_begin;
  for (Bytes at;; ++at) {
    bool at_span = at == span_begin || at == span_end;
    bool rewritten = at < size && !IsPrintable(charset, line[at.count()]);
    if (at < size && !at_span && !rewritten) {
      continue;
    }

    // Flush the run of bytes drawn as themselves. Measuring the run rather than
    // everything so far is what keeps this linear, and is valid because a run
    // only ever ends on a byte that starts a symbol.
    llvm::StringRef run =
        line.substr(run_begin.count(), (at - run_begin).count());
    result.text.append(run.begin(), run.end());
    cursor += metrics.Width(run);
    run_begin = at;

    if (at == span_begin) {
      start_column = cursor;
    }
    if (at == span_end) {
      end_column = cursor;
    }
    if (at == size) {
      break;
    }
    if (!rewritten) {
      continue;
    }

    if (line[at.count()] == '\t') {
      int width = SourceTabWidth - cursor % SourceTabWidth;
      result.text.append(width, ' ');
      cursor += width;
    } else {
      // TODO: Draw an escaped byte in a style of its own, such as a background
      // color, so that it can't be mistaken for the same characters written in
      // the source. This needs the styled runs that syntax highlighting does.
      auto value = static_cast<unsigned char>(line[at.count()]);
      result.text += '<';
      result.text += llvm::hexdigit(value >> 4);
      result.text += llvm::hexdigit(value & 0xf);
      result.text += '>';
      cursor += 4;
    }
    run_begin = at + Bytes(1);
  }

  result.span_start = start_column;
  result.span_length = std::max(end_column - start_column, 1);
  result.line_column = result.Middle();
  return result;
}

// Narrows `source` to `columns` around `focus`, with `WindowEllipsis` at each
// end that was elided.
//
// A source line and its underline that wrap onto different rows lose the
// alignment between them, which is the whole point of a snippet, so a line too
// wide for the terminal is elided rather than left to wrap.
//
// `focus` is the column the window is kept around, which is a span's own start
// unless several spans share one source row, in which case they must all be
// given the same one or their underlines land in different windows.
static auto WindowSource(Metrics metrics, PreparedSource& source, int columns,
                         int focus) -> void {
  // The width includes the column past the text that a span starting at the
  // end of the line is drawn in, so a line that exactly fills the columns is
  // still windowed to bring that underline inside them.
  int width = source.Width(metrics);
  if (width <= columns) {
    return;
  }

  // Both ellipses are accounted for whether or not both are needed, so that how
  // much source a window shows doesn't depend on where in the line it lands.
  auto ellipsis_width = static_cast<int>(WindowEllipsis.size());
  int inner = std::max(columns - 2 * ellipsis_width, MinWindowColumns);
  int begin = std::clamp(focus - inner / 4, 0, std::max(width - inner, 0));

  // The text is walked rather than indexed, so no byte offset into it is ever
  // held next to a column. What is dropped from the front is measured from what
  // was really taken, which can be a column short of what was asked for when a
  // double-width symbol straddles the boundary; measuring from the column asked
  // for would let the window keep that symbol's other column and come out a
  // column wider than it is allowed.
  llvm::StringRef rest = source.text;
  llvm::StringRef before = metrics.TakeColumns(rest, begin);
  int dropped = metrics.Width(before);
  llvm::StringRef window = metrics.TakeColumns(rest, inner);

  std::string text;
  int shift = 0;
  if (!before.empty()) {
    text.append(WindowEllipsis.begin(), WindowEllipsis.end());
    shift = ellipsis_width;
  }
  text.append(window.begin(), window.end());
  if (!rest.empty()) {
    text.append(WindowEllipsis.begin(), WindowEllipsis.end());
  }

  // Both ends of the span move by what was really dropped from in front of it,
  // and each is clamped into the window's own columns. Clamping the ends rather
  // than carrying the length is what keeps a span the window clipped short --
  // one windowed around another span on its row, off one edge -- from
  // underlining the source that took its place: what was clipped away is not
  // underlined. Every column is drawable and none is past the last cell, so the
  // underline stays an underline and never reaches past the grid.
  int visible = metrics.Width(text);
  int offset = shift - dropped;
  int old_end = source.span_start + source.span_length;
  source.span_start =
      std::clamp(source.span_start + offset, 0, std::min(visible, columns - 1));
  int span_end = std::clamp(old_end + offset, source.span_start + 1, columns);
  source.span_length = span_end - source.span_start;
  source.line_column = source.Middle();
  source.text = std::move(text);
}

// Returns line `number` of the file `part`'s source came from, or a null
// reference when there is no way to find it. A line that is genuinely empty
// still comes back as an empty reference over real data.
static auto LineFromFile(const PreparedPart& part, int number)
    -> llvm::StringRef {
  llvm::StringRef file = part.file_text;
  if (file.empty() || part.raw_line.empty() || number <= part.line_number) {
    return llvm::StringRef();
  }
  // `Loc::line` is meant to be a slice of `Loc::file_text`, which lets this
  // walk forward from it rather than counting from the top of the file. A `Loc`
  // is a plain aggregate that anything can fill in, so that is checked rather
  // than assumed.
  const char* line = part.raw_line.data();
  if (line < file.begin() || line >= file.end()) {
    return llvm::StringRef();
  }
  size_t offset = line - file.begin();
  for (auto _ : llvm::seq(part.line_number, number)) {
    size_t newline = file.find('\n', offset);
    if (newline == llvm::StringRef::npos) {
      return llvm::StringRef();
    }
    offset = newline + 1;
  }
  return file.substr(offset).take_until([](char c) { return c == '\n'; });
}

// Returns a part of a diagnostic prepared for drawing: `text` said at `loc`,
// drawn as `role` says and named `name` where kinds are being included.
//
// `is_leading` is set for the part that states the problem; see
// `PreparedPart::is_leading`.
static auto PreparePart(Metrics metrics, Part role, bool from_message_loc,
                        llvm::StringRef level_text, Style style, const Loc& loc,
                        std::string text, llvm::StringRef name, bool is_leading,
                        bool include_kind) -> PreparedPart {
  // Text past any legitimate size is truncated rather than handed to the
  // buffer, whose own limit on a single draw is a CHECK; nothing a diagnostic
  // says comes near either bound.
  if (text.size() > MaxDrawnBytes) {
    text.resize(MaxDrawnBytes);
  }

  PreparedPart result = {
      .role = role,
      .is_leading = is_leading,
      .from_message_loc = from_message_loc,
      .level_text = level_text,
      .style = style,
      .underline_style = UnderlineStyle(style),
      .location = FormatLocation(loc),
      .location_end_column = loc.column_number > 0 && loc.length > 1
                                 ? loc.column_number + loc.length - 1
                                 : 0,
      .file = loc.filename,
      .file_text = loc.file_text,
      .raw_line = loc.line,
      .line_number = loc.line_number};

  // Location information says how a place was reached rather than pointing at
  // code itself, so it never gets a snippet.
  if (role != Part::LocationInfo && loc.column_number > 0) {
    result.has_source = true;
    result.source = PrepareSource(metrics, loc.line, Bytes(loc.column_number),
                                  Bytes(loc.length));
  }

  // Text with no source to underline is placed under the part's location row.
  // Everything else, the message included, is a label placed under its
  // underline.
  if (!result.has_source) {
    result.text = std::move(text);
  } else {
    result.label = std::move(text);
  }

  // Every row is one row: text carrying a newline would otherwise be measured
  // as one column and drawn over the row below it.
  for (std::string* row : {&result.text, &result.label}) {
    for (char& c : *row) {
      if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
        c = ' ';
      }
    }
  }
  // A label that scrubbed away to spaces has nothing to hang, and hanging it
  // would spend rows and a label line on drawing nothing. It becomes what a
  // label with no words is: an underline and nothing else.
  if (result.label.find_first_not_of(' ') == std::string::npos) {
    result.label.clear();
  }

  if (include_kind && !name.empty()) {
    result.kind_suffix = (llvm::Twine(" [") + name + "]").str();
  }
  return result;
}

// Returns how `part`, which is location information, is spelled: what it says
// and then where, so that the words read as leading to the location. A part
// missing either half is just the other.
static auto PathText(const PreparedPart& part) -> std::string {
  if (part.location.empty()) {
    return part.text;
  }
  if (part.text.empty()) {
    return part.location;
  }
  return part.text + ": " + part.location;
}

// Returns whether `part` gets a location row: every part with a snippet to show
// or text to place under one, and the leading part, whose level word is
// always drawn. A range with no source and no text draws nothing.
static auto HasLocationRow(const PreparedPart& part) -> bool {
  return part.role != Part::LocationInfo &&
         (part.has_source || !part.text.empty() || part.is_leading);
}

// Returns whether a label line running down `column` can tee into the underline
// of `against` there.
//
// Whether a column offers a junction is asked of every underline on the row
// rather than of `against` alone, since a cell holds all of them together: a
// range drawn over this one covers what its own end gave up, and a label line
// meeting that reads as the tee it is drawn as. Asking only `against` would
// leave a label line running through a column another underline had filled,
// crossing it.
static auto TeesAt(const PreparedSource& against,
                   llvm::ArrayRef<PreparedPart*> spans, int column) -> bool {
  return against.Covers(column) &&
         llvm::any_of(spans,
                      [&](const PreparedPart* span) {
                        return span->source.ReachesLeftOf(column);
                      }) &&
         llvm::any_of(spans, [&](const PreparedPart* span) {
           return span->source.ReachesRightOf(column);
         });
}

// Returns where a label line asked to run down `desired` meets the underline of
// `against`, and the column it runs down.
//
// An end pulled in to clear the range beside it has no junction to offer, and
// a label line meeting one would form a corner, which looks like the underline
// turning down. So the label line steps a column away from that range, where
// the underlines run out through both sides again. Where no column of its range
// does -- an underline of one column that gave up an end -- the label line is
// drawn down that column in place of the underline. A wider underline never
// reaches this: an end gives way only where the range keeps a column a label
// line can tee into.
//
// `against` is the range the label line leaves: the label's own, or for the
// message, a range on the same row that contains the message's location.
static auto PlaceLabel(const PreparedSource& against,
                       llvm::ArrayRef<PreparedPart*> spans, int desired)
    -> Placement {
  if (TeesAt(against, spans, desired)) {
    return {Attachment::Tee, desired};
  }
  // Only an end fails to offer a junction, so the step is inward, away from
  // the range that pulled that end in.
  int step = desired == against.span_start ? 1 : -1;
  if (TeesAt(against, spans, desired + step)) {
    return {Attachment::Tee, desired + step};
  }
  // Nothing to tee into, so every label on this range takes the same column,
  // which is the one the underline is measured from rather than wherever the
  // words happened to have to start.
  return {Attachment::Through, against.Middle()};
}

// Draws the underline under `span`, in source drawn from column `x`, on row
// `y`.
//
// The heavy underline is drawn heavy in both character sets. ASCII draws a
// heavy line as `^`, so that underline stays distinct without color.
static auto DrawUnderline(Buffer& buffer, int x, int y,
                          const PreparedPart& span, const Style& style)
    -> void {
  // A label line with no column to tee into is drawn down the underline's own
  // column in place of it. Only an underline of one column is ever left with
  // none, so there is nothing of it to draw beside the line: an end gives up
  // its half column only where the range can still carry its label without it.
  if (span.source.label_line_stands_in) {
    return;
  }
  buffer.DrawHorizontalLine(
      x + span.source.span_start, y, span.source.span_length, style,
      span.source.left_end, span.source.right_end,
      span.is_heavy_underline ? Terminal::LineWeight::Heavy
                              : Terminal::LineWeight::Light);
}

// Returns the part whose range stands in for `message_part`'s own, or null when
// nothing does and the message's own location is underlined.
//
// A range attached as part of the problem says where the problem is in terms
// the reader can see, which the message's location -- a point, and often one
// chosen for an editor to put a cursor on rather than for anyone to look at --
// does not. So a range containing that location is underlined in its place: the
// message underlines nothing of its own, and its text is placed under that
// range.
//
// It has to contain the location. A range elsewhere, even on the same line, is
// not about the code the location names, and placing the message under it
// would drop the underline on the code the message is about.
//
// `candidates` are the ranges drawn alongside the message's, which is what a
// range has to be to take its place: one drawn under a location row of its own
// is read against the code below that one instead.
static auto FindMessageHost(llvm::ArrayRef<PreparedPart*> candidates,
                            const PreparedPart& message_part) -> PreparedPart* {
  int column = message_part.source.span_start;
  for (PreparedPart* candidate : candidates) {
    if (candidate == &message_part || candidate->role != Part::Problem ||
        candidate->file != message_part.file ||
        candidate->line_number != message_part.line_number) {
      continue;
    }
    const PreparedSource& source = candidate->source;
    if (column >= source.span_start &&
        column < source.span_start + source.span_length) {
      return candidate;
    }
  }
  return nullptr;
}

// Recolors the underlines of `spans`, which share one line, along their themes'
// ramps. Each theme counts its own way through its colors, left to right, so
// they run in the order the code they underline is read.
//
// The heavy underline is left out of the ramp: it keeps the level's center
// color and its bold, matching the level word. The other problem ranges on its
// line take successive colors from the ramp with the center left out, so that
// only the heavy underline has the center color.
static auto RotateUnderlineStyles(llvm::ArrayRef<PreparedPart*> spans,
                                  Theme problem_theme, Theme explanation_theme)
    -> void {
  llvm::SmallVector<PreparedPart*> across(spans);
  llvm::stable_sort(across,
                    [](const PreparedPart* lhs, const PreparedPart* rhs) {
                      return lhs->source.span_start < rhs->source.span_start;
                    });
  bool leading_on_line = llvm::any_of(across, [](const PreparedPart* span) {
    return span->is_heavy_underline;
  });
  llvm::SmallVector<Terminal::Color> around_center;
  if (leading_on_line) {
    for (auto [index, color] : llvm::enumerate(problem_theme)) {
      if (static_cast<int>(index) != ThemeCenter) {
        around_center.push_back(color);
      }
    }
  }
  int totals[2] = {0, 0};
  for (const PreparedPart* span : across) {
    totals[span->role == Part::Problem] += !span->is_heavy_underline;
  }
  int counts[2] = {0, 0};
  for (PreparedPart* span : across) {
    if (span->is_heavy_underline) {
      span->underline_style = span->style;
      continue;
    }
    bool is_problem = span->role == Part::Problem;
    int& count = counts[is_problem];
    if (is_problem && leading_on_line) {
      span->underline_style = span->underline_style.Foreground(
          around_center[count++ % around_center.size()]);
      continue;
    }
    span->underline_style = RotatedStyle(
        span->underline_style, is_problem ? problem_theme : explanation_theme,
        count++, totals[is_problem]);
  }
}

// Pulls in the ends of `spans` that touch another span on their line.
//
// Two ranges that touch have nothing between them to say where one ends and
// the next begins, so an end that meets one stops at the center of its cell
// rather than running out through the side. That leaves half a column of gap,
// and a full one where both ends give way.
//
// The gap is worth having but not worth paying for. An end gives way only
// where the range can still carry its label without it: two columns can spare
// one end but not both, since a label line needs a column the underline runs
// out through on both sides and after the left end gave way only the right one
// is left. Left is considered first, so the range that begins at a boundary is
// always the one that gives way there, and every boundary ends up with a gap.
// One column can spare neither end, and the accepted answer there is the label
// line drawn down that column, or a point where it has no label line.
static auto SeparateTouchingEnds(
    llvm::ArrayRef<PreparedPart*> spans,
    const llvm::SmallPtrSetImpl<const PreparedPart*>& carries_label) -> void {
  for (PreparedPart* span : spans) {
    int span_end = span->source.span_start + span->source.span_length;
    auto meets = [&](auto&& touches) {
      return llvm::any_of(spans, [&](const PreparedPart* other) {
        return other != span && touches(other->source);
      });
    };
    if (meets([&](const PreparedSource& other) {
          return other.span_start + other.span_length ==
                 span->source.span_start;
        })) {
      span->source.left_end = LineEnd::Center;
    }
    if (meets([&](const PreparedSource& other) {
          return other.span_start == span_end;
        })) {
      span->source.right_end = LineEnd::Center;
      if (carries_label.contains(span) && span->source.span_length > 1 &&
          !span->source.Joinable()) {
        span->source.right_end = LineEnd::Edge;
      }
    }
  }
}

// Returns the range `label` is placed under: the message host's for the
// message's own part when a host has taken its place, and otherwise its own.
static auto LabelSource(const PreparedPart* label,
                        const PreparedPart* message_part,
                        const PreparedPart* message_host)
    -> const PreparedSource& {
  return label == message_part && message_host ? message_host->source
                                               : label->source;
}

// Sets the column each label line runs down, choosing for the whole line at
// once. `labels` are in reading order, which the loop assigning each label its
// step relies on.
//
// Each label line starts at the middle of its underline, so that it points into
// the range. Every label's text starts right of every label line's column, so
// when the text on the line needs more room than the rightmost label lines
// leave, each label line right of where the text has to start slides left,
// whichever label it belongs to: only as far as the text needs to fit on one
// row, and never past where its own range starts. Label lines slide before any
// text wraps, and the label line of a label that will be out-dented still
// slides for the others.
//
// `message_part` and `message_host` are as in `RowLayout::AddLine`: the
// message's text is placed under the host's range when one has taken its
// place.
static auto PlaceLabels(const Buffer& buffer,
                        llvm::ArrayRef<PreparedPart*> labels,
                        const PreparedPart* message_part,
                        const PreparedPart* message_host,
                        llvm::ArrayRef<PreparedPart*> spans, int content_x,
                        int columns) -> void {
  Metrics metrics = buffer.metrics();
  auto against = [&](const PreparedPart* label) -> const PreparedSource& {
    return LabelSource(label, message_part, message_host);
  };
  // The leftmost column each label line can run down: where its range starts,
  // or the next column when the start has no junction, and the middle when
  // neither has one.
  auto floor_of = [&](const PreparedSource& source) {
    for (int column : {source.span_start, source.span_start + 1}) {
      if (TeesAt(source, spans, column)) {
        return column;
      }
    }
    return source.Middle();
  };

  // The base is the column the first label's text starts in. These are the
  // base with every label line at its middle, and the lowest base, with every
  // label line at its floor.
  int natural_base = 0;
  int lowest_base = 0;
  for (const PreparedPart* label : labels) {
    natural_base =
        std::max(natural_base, against(label).Middle() + LabelIndentWidth);
    lowest_base =
        std::max(lowest_base, floor_of(against(label)) + LabelIndentWidth);
  }

  // The most columns right of the base that any label needs: its step, level
  // word, and text on one row. A label whose widest word does not fit even
  // from the lowest base will be out-dented, so its text needs nothing of the
  // base, though its label line still slides with the rest.
  int width_limit = columns - content_x;
  int worst = 0;
  bool any_fits = false;
  int step = 0;
  for (const PreparedPart* label : labels) {
    int prefix_width = label->is_leading ? metrics.Width(label->level_text) : 0;
    if (lowest_base + step + prefix_width +
            buffer.MeasureWrapWidth(label->label, BackticksAreQuotes) >
        width_limit) {
      continue;
    }
    worst = std::max(worst, step + prefix_width + metrics.Width(label->label));
    any_fits = true;
    step += LabelStepWidth;
  }
  int target_base = natural_base;
  if (any_fits) {
    target_base =
        std::max(lowest_base, std::min(width_limit - worst, natural_base));
  }

  for (PreparedPart* label : labels) {
    const PreparedSource& source = against(label);
    int desired = std::clamp(target_base - LabelIndentWidth, floor_of(source),
                             source.Middle());
    Placement placement = PlaceLabel(source, spans, desired);
    label->source.line_column = placement.line_column;
    // A label line drawn in place of an underline is recorded on the range it
    // stands in for, not on the label that asked for it -- and on every range
    // drawn over exactly the same column, since a second underline there would
    // be a half-cell line beside the label line rather than a line through it,
    // which is the one thing that would leave a corner in the cell.
    if (placement.attachment == Attachment::Through) {
      for (PreparedPart* span : spans) {
        if (span->source.span_start == source.span_start &&
            span->source.span_length == source.span_length) {
          span->source.label_line_stands_in = true;
        }
      }
    }
  }
}

// Returns the spans on `line` that draw an underline: all of them, except
// `message_part` once `message_host` has taken its place.
//
// Every underline for the line goes on the one row below it, so that the
// underlines read as underlining that line rather than as a stack of rows each
// of which has to be traced back up to it. Ranges that overlap are drawn widest
// first, so the narrowest underline covering a column is the one that shows: a
// range containing another says less about the columns they share than the
// range inside it does, and an underline that is part of the problem says more
// than one that only explains it. The heavy underline is the exception; see
// `MoveHeavyUnderlinesLast`.
//
// Ranges shouldn't overlap in the first place -- one covering another means two
// parts of the diagnostic attached to the same code -- but they are attached by
// layers that never see each other, so this repairs what it is given rather
// than checking a rule nothing is in a position to enforce.
static auto SpansWidestFirst(llvm::ArrayRef<PreparedPart*> line,
                             const PreparedPart* message_part,
                             const PreparedPart* message_host)
    -> llvm::SmallVector<PreparedPart*> {
  llvm::SmallVector<PreparedPart*> spans;
  for (PreparedPart* member : line) {
    if (member != message_part || !message_host) {
      spans.push_back(member);
    }
  }
  llvm::stable_sort(
      spans, [](const PreparedPart* lhs, const PreparedPart* rhs) {
        if (lhs->source.span_length != rhs->source.span_length) {
          return lhs->source.span_length > rhs->source.span_length;
        }
        return lhs->role < rhs->role;
      });
  return spans;
}

// Sets `is_heavy_underline` on the spans drawn as the heavy underline and moves
// them to the end of `spans`. Returns the span the leading part's text is
// placed under, or null when that text is not placed on this line.
static auto MoveHeavyUnderlinesLast(llvm::MutableArrayRef<PreparedPart*> spans,
                                    const PreparedPart* message_part,
                                    const PreparedPart* message_host)
    -> const PreparedPart* {
  // The leading part's text is placed under its own range, or under the host
  // of the message's location when the message leads.
  const PreparedPart* leading_host =
      message_part && message_part->is_leading ? message_host : nullptr;
  const PreparedPart* heavy_underline = nullptr;
  for (PreparedPart* span : spans) {
    span->is_heavy_underline = span->is_leading || span == leading_host;
    if (span->is_heavy_underline) {
      heavy_underline = span;
    }
  }
  // A problem range identical to the leading one underlines the same source.
  // Identical ranges are drawn over each other, so it takes the heavy
  // underline's weight and color; otherwise it would repaint that stretch
  // light, in a rotated color.
  if (heavy_underline) {
    for (PreparedPart* span : spans) {
      span->is_heavy_underline |=
          span->role == Part::Problem &&
          span->source.span_start == heavy_underline->source.span_start &&
          span->source.span_length == heavy_underline->source.span_length;
    }
  }
  // The heavy underline is drawn last, over every other range on the row, so
  // that no narrower range redraws a cell of it light. A range inside it shows
  // as the tee where its label line leaves the heavy underline.
  std::stable_partition(
      spans.begin(), spans.end(),
      [](const PreparedPart* span) { return !span->is_heavy_underline; });
  return heavy_underline;
}

// Returns the parts whose words hang off `spans`, in the order of `spans`.
static auto LabelsHangingFrom(llvm::ArrayRef<PreparedPart*> spans,
                              PreparedPart* message_part,
                              const PreparedPart* message_host)
    -> llvm::SmallVector<PreparedPart*> {
  llvm::SmallVector<PreparedPart*> labels;
  for (PreparedPart* span : spans) {
    // The host of the part's location has the part's text placed under it as
    // well as its own, and the part's text comes first, so the sentence
    // stating the problem is next to the underline with the range's own text
    // below.
    if (span == message_host && !message_part->label.empty()) {
      labels.push_back(message_part);
    }
    if (!span->label.empty()) {
      labels.push_back(span);
    }
  }
  return labels;
}

// Sorts `labels` into the order their rows are drawn in.
//
// The rows read top to bottom the way the code reads left to right: the
// leading part first, then each label in the order of the middle of its
// range. Ordering by the range, not the label line, keeps a sliding label line
// from reordering the rows. A label line that descends through rows drawn above
// it breaks around what they drew.
static auto SortLabelsForReading(llvm::MutableArrayRef<PreparedPart*> labels,
                                 const PreparedPart* message_part,
                                 const PreparedPart* message_host) -> void {
  auto order_key = [&](const PreparedPart* label) {
    return LabelSource(label, message_part, message_host).Middle();
  };
  llvm::stable_sort(labels,
                    [&](const PreparedPart* lhs, const PreparedPart* rhs) {
                      if (lhs->is_leading != rhs->is_leading) {
                        return lhs->is_leading;
                      }
                      return order_key(lhs) < order_key(rhs);
                    });
}

namespace {

// Lays out the rows a diagnostic's frame holds, in the order they are drawn.
//
// `source_columns` bounds every source row: the lines shown between two spans
// as well as the spans' own, which are windowed here so that spans sharing a
// row share one view of their line. `buffer` is the one the rows will be drawn
// into, and is only measured against here: how far text wraps is its answer to
// give.
class RowLayout {
 public:
  RowLayout(const Buffer& buffer, llvm::MutableArrayRef<PreparedPart> parts,
            int leading, Theme problem_theme, Theme explanation_theme,
            int source_columns, int content_x, int columns);

  // Returns the rows, in the order they are drawn.
  auto LayOut() && -> llvm::SmallVector<Row>;

 private:
  // Adds the rows for `parts_[index]`: the path to its location, the location
  // row naming it, its text when it has no source, and the snippet below that,
  // which also shows the spans of the parts `TakeGroup` gathers with it.
  auto AddBlock(size_t index) -> void;

  // Returns the spans shown under the location row of `parts_[index]`: its own
  // and every later span in the same file, shown as one view of it in source
  // order with the lines between them elided. A part that names only a file has
  // no span of its own, but its location row is still the one the file's spans
  // are shown under, so that the file doesn't get a second location row for the
  // first of them. Records the later spans as drawn.
  auto TakeGroup(size_t index) -> llvm::SmallVector<PreparedPart*>;

  // Adds the rows showing `group`'s source below its location row, one line of
  // source at a time.
  auto AddSnippet(llvm::ArrayRef<PreparedPart*> group,
                  PreparedPart* message_part, PreparedPart* message_host)
      -> void;

  // Adds the lines of source between `previous`'s line and line `next`, or an
  // elision standing in for them.
  auto AddLinesBetween(const PreparedPart& previous, int next) -> void;

  // Adds the rows drawing the spans that fall on one source line: the line
  // itself, the underline row beneath it, and the label rows under that.
  //
  // `message_part` is the range standing in for the part's own location, and
  // `message_host` the attached range that has taken its place, in which case
  // the part underlines nothing of its own and its words hang from the host.
  auto AddLine(llvm::ArrayRef<PreparedPart*> line, PreparedPart* message_part,
               PreparedPart* message_host) -> void;

  // Adds a row for each of `labels`, in order, below their underline row, and
  // sets where each label's text starts.
  auto AddLabelRows(llvm::ArrayRef<PreparedPart*> labels) -> void;

  // Returns whether the last row added was below a location row, so that the
  // step to the next location row gets an elision.
  auto InSnippet() const -> bool;

  const Buffer* buffer_;
  Metrics metrics_;
  llvm::MutableArrayRef<PreparedPart> parts_;
  int leading_;
  Theme problem_theme_;
  Theme explanation_theme_;
  int source_columns_;
  int content_x_;
  int columns_;

  // Whether a part has to open a block of its own rather than being merged
  // into an earlier one: its text is placed under a location row of its own, or
  // there is location information above it that needs its location row to lead
  // into.
  llvm::SmallVector<bool> opens_block_;

  // Whether each part's rows have been added.
  llvm::SmallVector<bool> drawn_;

  // Location information waiting for the location row it describes.
  llvm::SmallVector<const PreparedPart*> path_;

  llvm::SmallVector<Row> rows_;
};

}  // namespace

RowLayout::RowLayout(const Buffer& buffer,
                     llvm::MutableArrayRef<PreparedPart> parts, int leading,
                     Theme problem_theme, Theme explanation_theme,
                     int source_columns, int content_x, int columns)
    : buffer_(&buffer),
      metrics_(buffer.metrics()),
      parts_(parts),
      leading_(leading),
      problem_theme_(problem_theme),
      explanation_theme_(explanation_theme),
      source_columns_(source_columns),
      content_x_(content_x),
      columns_(columns),
      opens_block_(parts.size(), false),
      drawn_(parts.size(), false) {
  bool path_pending = false;
  for (auto [index, part] : llvm::enumerate(parts)) {
    if (part.role == Part::LocationInfo) {
      path_pending = true;
      continue;
    }
    opens_block_[index] = path_pending || !part.text.empty();
    path_pending = false;
  }
}

auto RowLayout::LayOut() && -> llvm::SmallVector<Row> {
  for (size_t index : llvm::seq(parts_.size())) {
    if (!drawn_[index]) {
      AddBlock(index);
    }
  }
  CARBON_CHECK(path_.empty(), "Location information precedes its part.");
  return std::move(rows_);
}

auto RowLayout::AddBlock(size_t index) -> void {
  drawn_[index] = true;
  PreparedPart& part = parts_[index];

  // Location information says how the location after it was reached, so it
  // waits and is then drawn above the location row for that location.
  if (part.role == Part::LocationInfo) {
    path_.push_back(&part);
    return;
  }

  // A part with no source and no text draws nothing, so the path waiting for
  // its location row is dropped.
  if (!HasLocationRow(part)) {
    path_.clear();
    return;
  }

  if (InSnippet()) {
    rows_.push_back({.kind = Row::Kind::Elision});
  }

  for (const PreparedPart* reached_from : path_) {
    rows_.push_back({.kind = Row::Kind::Path, .part = reached_from});
  }
  path_.clear();

  llvm::SmallVector<PreparedPart*> group = TakeGroup(index);
  rows_.push_back({.kind = Row::Kind::Location, .part = &part});

  // A part with no source has its text placed under its location row. The
  // file's spans, if any, follow below an elision, since they are elsewhere in
  // the file.
  //
  // TODO: No toolchain diagnostic should reach this: the goal is every part
  // attached to a location that reaches source, with one about a file as a
  // whole located at the file's start or end and drawn in a form of its own.
  // That restructuring starts with the emitting diagnostics.
  if (!part.has_source) {
    rows_.push_back(
        {.kind = Row::Kind::LocationLabel, .part = &part, .text = part.text});
    if (!group.empty()) {
      rows_.push_back({.kind = Row::Kind::Elision});
    }
  }

  // The spans shown under one location row are the ones drawn alongside each
  // other here, so the group is what a range has to be in to stand in for the
  // part's location: a range under a location row of its own is read against
  // the code below that one instead.
  PreparedPart* message_part = nullptr;
  for (PreparedPart* member : group) {
    if (member->from_message_loc) {
      message_part = member;
      break;
    }
  }
  PreparedPart* message_host =
      message_part ? FindMessageHost(group, *message_part) : nullptr;
  if (message_host) {
    // The part underlines nothing of its own once something else has taken its
    // place, so the line it named is not part of the snippet either -- unless
    // an attached range is on it too. Leaving it in shows a line of source with
    // nothing underlined, which reads as a line the reader is meant to find
    // something in.
    llvm::erase(group, message_part);
  }

  AddSnippet(group, message_part, message_host);
}

auto RowLayout::TakeGroup(size_t index) -> llvm::SmallVector<PreparedPart*> {
  const PreparedPart& part = parts_[index];
  llvm::SmallVector<PreparedPart*> group;
  if (part.has_source) {
    group.push_back(&parts_[index]);
  } else if (part.file.empty()) {
    return group;
  }
  for (size_t other : llvm::seq(index + 1, parts_.size())) {
    if (!drawn_[other] && !opens_block_[other] && parts_[other].has_source &&
        parts_[other].file == part.file) {
      drawn_[other] = true;
      group.push_back(&parts_[other]);
    }
  }
  llvm::stable_sort(group,
                    [](const PreparedPart* lhs, const PreparedPart* rhs) {
                      return lhs->line_number < rhs->line_number;
                    });
  return group;
}

auto RowLayout::AddSnippet(llvm::ArrayRef<PreparedPart*> group,
                           PreparedPart* message_part,
                           PreparedPart* message_host) -> void {
  // Spans on one line are drawn together, against the one row that shows it.
  const PreparedPart* previous = nullptr;
  for (size_t begin = 0; begin < group.size();) {
    size_t end = begin + 1;
    while (end < group.size() &&
           group[end]->line_number == group[begin]->line_number) {
      ++end;
    }
    if (previous) {
      AddLinesBetween(*previous, group[begin]->line_number);
    }
    AddLine(group.slice(begin, end - begin), message_part, message_host);
    previous = group[end - 1];
    begin = end;
  }
}

auto RowLayout::AddLinesBetween(const PreparedPart& previous, int next)
    -> void {
  if (next - previous.line_number - 1 > MaxShownGap) {
    rows_.push_back({.kind = Row::Kind::Elision});
    return;
  }
  for (int number : llvm::seq(previous.line_number + 1, next)) {
    llvm::StringRef line = LineFromFile(previous, number);
    if (line.data() == nullptr) {
      // Without the file there is no way to show what was skipped, and a blank
      // row would claim the line is empty.
      rows_.push_back({.kind = Row::Kind::Elision});
      continue;
    }
    PreparedSource shown = PrepareSource(metrics_, line, Bytes(1), Bytes(1));
    WindowSource(metrics_, shown, source_columns_, /*focus=*/0);
    rows_.push_back({.kind = Row::Kind::Source,
                     .line_number = number,
                     .text = std::move(shown.text)});
  }
}

auto RowLayout::AddLine(llvm::ArrayRef<PreparedPart*> line,
                        PreparedPart* message_part, PreparedPart* message_host)
    -> void {
  // Spans sharing the row are windowed around the first of them: they share
  // the one source row that shows their line, so windowing each around its
  // own span would land their underlines in different views of it. Spans
  // drawn on different rows never share a window, wherever in the file they
  // are.
  int focus = line.front()->source.span_start;
  for (PreparedPart* member : line) {
    WindowSource(metrics_, member->source, source_columns_, focus);
  }

  llvm::SmallVector<PreparedPart*> spans =
      SpansWidestFirst(line, message_part, message_host);
  const PreparedPart* heavy_underline =
      MoveHeavyUnderlinesLast(spans, message_part, message_host);
  RotateUnderlineStyles(spans, problem_theme_, explanation_theme_);
  // The part's label line and bar follow the range hosting its words, so the
  // rotation's choice for the host is copied after it is made.
  if (message_host && llvm::is_contained(spans, message_host)) {
    message_part->underline_style = message_host->underline_style;
  }

  // The `->` in the margin points to the one line to begin reading at: the line
  // the leading part's text is placed under, wherever it is in the frame. A
  // problem range on any other line keeps its color and nothing more. The
  // `->` and line number take the leading part's style, matching its level
  // word.
  rows_.push_back({.kind = Row::Kind::Source,
                   .part = line.front(),
                   .points_here = heavy_underline ? &parts_[leading_] : nullptr,
                   .line_number = line.front()->line_number,
                   .text = line.front()->source.text});

  llvm::SmallVector<PreparedPart*> labels =
      LabelsHangingFrom(spans, message_part, message_host);

  // The ranges a label line will hang from, which is what decides whether a
  // range can afford to give up half a column at an end.
  llvm::SmallPtrSet<const PreparedPart*, 4> carries_label;
  for (PreparedPart* label : labels) {
    carries_label.insert(label == message_part && message_host ? message_host
                                                               : label);
  }

  SeparateTouchingEnds(spans, carries_label);
  SortLabelsForReading(labels, message_part, message_host);
  PlaceLabels(*buffer_, labels, message_part, message_host, spans, content_x_,
              columns_);

  rows_.push_back(
      {.kind = Row::Kind::Underline, .spans = {spans.begin(), spans.end()}});
  AddLabelRows(labels);
}

auto RowLayout::AddLabelRows(llvm::ArrayRef<PreparedPart*> labels) -> void {
  // The leftmost column the next label's text may start in, stepping right
  // with each label so the indentation follows the row order. It starts right
  // of every label line's column, so a label line descending to a later label
  // runs beside the text above it, crossing each row in a single cell of a
  // run and never at a junction.
  int next_label_x = 0;
  for (const PreparedPart* label : labels) {
    next_label_x =
        std::max(next_label_x, label->source.line_column + LabelIndentWidth);
  }
  for (PreparedPart* label : labels) {
    label->source.label_x =
        std::max(next_label_x, label->source.line_column + LabelIndentWidth);
    // A single unbreakable word can be wider than what is left to the right
    // of where the label hangs, and hanging it there would push it out of the
    // frame. Wrapping never breaks a word, so the widest one is the least
    // width the label can be drawn in. Below that the label is out-dented to
    // the column the source starts in, where the whole width is available,
    // and the line reaching back to the label line is what says which range it
    // belongs to.
    int prefix_width =
        label->is_leading ? metrics_.Width(label->level_text) : 0;
    bool routed =
        content_x_ + label->source.label_x + prefix_width +
            buffer_->MeasureWrapWidth(label->label, BackticksAreQuotes) >
        columns_;
    // An out-dented label's text starts at the source's column, so it does
    // not move where the next label's text starts.
    if (!routed) {
      next_label_x = label->source.label_x + LabelStepWidth;
    }
    rows_.push_back({.kind = routed ? Row::Kind::RoutedLabel : Row::Kind::Label,
                     .part = label,
                     .text = label->label});
  }
}

auto RowLayout::InSnippet() const -> bool {
  if (rows_.empty()) {
    return false;
  }
  switch (rows_.back().kind) {
    case Row::Kind::Source:
    case Row::Kind::Underline:
    case Row::Kind::Label:
    case Row::Kind::RoutedLabel:
    case Row::Kind::LocationLabel:
    case Row::Kind::Elision:
      return true;
    case Row::Kind::Location:
    case Row::Kind::Path:
      return false;
  }
}

// Draws the leading part's label line, from the underline at `underline_y` down
// to `meet_y`.
//
// It stops at the center of the cell the underline drew rather than redrawing
// it, so where the underline runs out through both sides of that cell the two
// form a junction. Where it stands in for an underline of one column instead,
// stopping at the center is what keeps the underline to the lower half of its
// cell, so that two of them in neighboring columns read as two underlines
// rather than as one line drawn down both.
//
// The cell on the underline row is drawn heavy whatever `weight` is, since it
// is part of the heavy underline. Under ASCII, where `weight` is light, that
// cell keeps the tee inside the `^` run, or is the `^` itself when the label
// line is drawn in place of a one-column underline.
static auto DrawLeadingLabelLine(Buffer& buffer, int x, int underline_y,
                                 int meet_y, const Style& style,
                                 Terminal::LineWeight weight) -> void {
  buffer.DrawVerticalLine(x, underline_y, 1, style, LineEnd::Center,
                          LineEnd::Edge, Terminal::LineWeight::Heavy);
  buffer.DrawVerticalLine(x, underline_y + 1, meet_y - underline_y, style,
                          LineEnd::Edge, LineEnd::Center, weight);
}

// Draws `number` in the columns before the frame, or nothing when there is no
// line to name.
//
// `points_here` is the leading part when its text is placed under this line.
// It draws `->` in the margin and colors the number to match the level word.
// Every other number keeps the frame's color, so the line to begin reading at
// is the one that stands out from the frame.
static auto DrawLineNumber(Buffer& buffer, int frame_x, int y, int number,
                           const PreparedPart* points_here) -> void {
  if (number <= 0) {
    return;
  }
  std::string text = std::to_string(number);
  // The line the problem is on takes the leading part's color and weight, and
  // every other line number is left as light as the frame it belongs to, so
  // that the one being reported is the one that stands out rather than every
  // number competing with it.
  Style style = points_here ? points_here->style : FrameStyle().Bold(false);
  buffer.DrawText(frame_x - 1 - text.size(), y, text, style);
  if (points_here) {
    buffer.DrawText(0, y, LineArrow, style);
  }
}

// Returns whether `text` and `suffix`, wrapped together into `columns` at
// (x, y), fit on the one row they start on.
//
// Measuring the two together, rather than asking whether the suffix fits
// after the text, is what gives a block that wrapped at all a tag naming it
// rather than trailing its last word.
static auto WrappedSharesRow(const Buffer& buffer, int x, int y, int columns,
                             llvm::StringRef text, llvm::StringRef suffix)
    -> bool {
  return buffer
             .MeasureWrappedText(x, y, /*margin=*/x, columns,
                                 (llvm::Twine(text) + suffix).str(),
                                 BackticksAreQuotes)
             .y == y;
}

// Draws `text` wrapped into `columns` at (x, y), with `suffix` after it, and
// returns the row after what it drew.
//
// The suffix is drawn on its own so that it keeps its own style, and placed
// where wrapping would have put it: after the text where the two share a row,
// and on a row of its own at the margin where they do not.
static auto DrawWrapped(Buffer& buffer, int x, int y, int columns,
                        llvm::StringRef text, llvm::StringRef suffix,
                        Style style = Style()) -> int {
  if (suffix.empty()) {
    return buffer
               .DrawWrappedText(x, y, /*margin=*/x, columns, text, style,
                                BackticksAreQuotes)
               .y +
           1;
  }
  Buffer::DrawEnd end = buffer.DrawWrappedText(x, y, /*margin=*/x, columns,
                                               text, style, BackticksAreQuotes);
  if (WrappedSharesRow(buffer, x, y, columns, text, suffix)) {
    buffer.DrawText(end.x, end.y, suffix, KindStyle());
    return end.y + 1;
  }
  // The separating space would indent a tag that starts a row.
  buffer.DrawText(x, end.y + 1, suffix.ltrim(), KindStyle());
  return end.y + 2;
}

auto Renderer::Render(Terminal::OutputBufferRef out,
                      const Diagnostic& diagnostic) const -> void {
  // Widths depend only on the character set, so this measures without ever
  // drawing into anything.
  Metrics metrics(capabilities_.charset);

  // A context names the operation that failed, and leads the diagnostic when
  // there is one: it is the failure the reader is being told about, and the
  // message is then read against the code like anything else explaining it.
  const Context* leading_context = LeadingContext(diagnostic);

  // A `PreparedPart` is undesirably large for inline storage by
  // SmallVector, so we specify 0.
  llvm::SmallVector<PreparedPart, 0> parts;
  int leading = -1;

  // Which colors this terminal gets, decided once: everything drawn takes one
  // of them.
  Theme problem_theme = LevelTheme(diagnostic.level, capabilities_);
  Theme explanation_theme = ExplanationTheme(capabilities_);

  // What leads the diagnostic is drawn at its level and everything else as
  // something explaining it, except that a primary range is part of the problem
  // and so keeps the level's color wherever down the frame it sits.
  auto add_part = [&](bool is_leading, bool is_problem, bool from_message_loc,
                      const Loc& loc, std::string text, llvm::StringRef name) {
    if (is_leading) {
      leading = parts.size();
    }
    parts.push_back(
        PreparePart(metrics, is_problem ? Part::Problem : Part::Explanation,
                    from_message_loc,
                    is_leading ? LevelText(diagnostic.level) : ExplanationText,
                    ThemeStyle(is_problem ? problem_theme : explanation_theme),
                    loc, std::move(text), name, is_leading, include_kind_));
  };

  // The same file reached the same way is a path drawn once. A path says how a
  // file was got to, so every range underlined in that file has the same one,
  // and repeating it under each of them says nothing the first didn't.
  std::string last_place;
  auto add_location_info = [&](llvm::ArrayRef<LocationInfo> steps,
                               const Loc& reaches) {
    std::string place = reaches.filename.str();
    for (const LocationInfo& step : steps) {
      place += "\n" + step.Format() + " " + FormatLocation(step.loc);
    }
    if (place == last_place) {
      return;
    }
    last_place = std::move(place);
    for (const LocationInfo& step : steps) {
      parts.push_back(PreparePart(
          metrics, Part::LocationInfo, /*from_message_loc=*/false,
          /*level_text=*/"", PathStyle(), step.loc, step.Format(), step.name,
          /*is_leading=*/false, include_kind_));
    }
  };

  // The contexts, then the message, then everything attached to explain it,
  // which is the order they are read in.
  for (const Context& context : diagnostic.contexts) {
    add_location_info(context.location_info, context.loc);
    // Only the leading context is the failure being reported; the rest name
    // operations it happened inside, which explain it.
    bool is_leading = &context == leading_context;
    add_part(is_leading, is_leading, /*from_message_loc=*/false, context.loc,
             context.Format(), context.name);
  }
  const Message& message = diagnostic.message;
  add_location_info(message.location_info, message.loc);
  // The message states the problem whether or not a context leads it: a
  // context gives it a sentence to sit under, and doesn't demote what it says
  // to an explanation of something else. Doing otherwise puts an informational
  // label line on a range that is underlined as part of the problem.
  bool message_leads = leading_context == nullptr;
  add_part(message_leads, /*is_problem=*/true, /*from_message_loc=*/true,
           message.loc, message.Format(), message.kind.name());
  // A label's location information is not drawn: how the file holding a
  // declaration was reached is rarely useful, and Clang and GCC omit note
  // include stacks for the same reason. Only the path to the problem itself,
  // the message's own location and a context's, is shown.
  for (const Label& label : diagnostic.labels) {
    add_part(/*is_leading=*/false, label.category == LabelCategory::Primary,
             /*from_message_loc=*/false, label.loc, label.Format(), label.name);
  }
  CARBON_CHECK(leading >= 0, "Either the context or the message leads.");

  // With snippets off, every diagnostic takes the compact form: nothing below
  // is drawn, so none of its geometry needs deciding.
  if (!snippets_) {
    RenderCompact(out, parts);
    return;
  }

  // Where content starts, which is what a width has to hold on top of the
  // source it shows. Line numbers sit to the left of the frame, so their width
  // is what decides which column it runs down; any line shown between two spans
  // has a number between theirs, so the widest is always one of these.
  int frame_x = MarginWidth + LineNumberColumns + 1;
  for (const PreparedPart& part : parts) {
    if (part.has_source && part.line_number > 0) {
      frame_x = std::max(
          frame_x,
          MarginWidth + metrics.Width(std::to_string(part.line_number)) + 1);
    }
  }
  int content_x = frame_x + FrameToContentWidth;

  // What the gutter costs when a line number fits the columns set aside for it,
  // which is what the target width is built on. A wider number spends the
  // difference out of the source rather than out of the width.
  int target_content_x =
      MarginWidth + LineNumberColumns + 1 + FrameToContentWidth;

  // The width every layout decision below is made against, taken from the
  // capabilities when they hold one. `COLUMNS` is whatever a user exported, so
  // that width is held to what a grid can hold before anything is laid out
  // against it. When there is none to take, the target is the source width
  // plus whatever this diagnostic spends reaching it, so the gutter is paid
  // for out of the total rather than out of the code.
  int columns = capabilities_.columns
                    ? std::min(*capabilities_.columns, Buffer::MaxColumns)
                    : target_content_x + FormattedSourceColumns;

  // What is left for source once the frame and its gutter are paid for. This
  // has to happen before the rows take their copy of the source they show.
  int source_columns = std::max(columns - content_x, 1);

  // The frame exists to hold source, so a diagnostic with none to show -- one
  // for a file that couldn't be opened, say -- gets the compact form instead of
  // an empty frame around a location. So does a terminal too narrow to hold a
  // frame, a gutter, and a label hanging off a span: below that the compact
  // form carries more in the columns there are.
  bool has_source = llvm::any_of(
      parts, [](const PreparedPart& part) { return part.has_source; });
  if (!has_source || columns < MinFrameColumns) {
    RenderCompact(out, parts);
    return;
  }

  RenderFramed(out, parts, leading, diagnostic.level, frame_x, content_x,
               source_columns, columns);
}

namespace {

// Draws the rows `RowLayout` lays out, with the frame beside them.
class FrameDrawer {
 public:
  FrameDrawer(Buffer& buffer, Charset charset, int frame_x, int content_x,
              int columns, Style dotted_frame_style)
      : buffer_(&buffer),
        glyphs_(Glyphs::For(charset)),
        charset_(charset),
        frame_x_(frame_x),
        content_x_(content_x),
        columns_(columns),
        dotted_frame_style_(dotted_frame_style) {}

  // Draws `rows` from the buffer's first row, followed by the closing rule.
  auto Draw(llvm::ArrayRef<Row> rows) -> void;

 private:
  // Each of these draws one kind of row starting at row `y`, and returns the
  // row after what it drew, which is further down when text wraps.
  auto DrawLocationRow(const PreparedPart& part, int y) -> int;
  auto DrawPath(const PreparedPart& part, int y) -> int;
  auto DrawSource(const Row& row, int y) -> int;
  auto DrawUnderlineRow(const Row& row, int y) -> int;
  auto DrawLabel(const Row& row, int y) -> int;
  auto DrawRoutedLabel(const Row& row, int y) -> int;
  auto DrawLocationLabel(const Row& row, int y) -> int;

  // Draws the frame beside rows `[y, next_y)`, which hold a row of `kind`.
  auto DrawFrame(Row::Kind kind, int y, int next_y) -> void;

  // Returns the stroke weight of the label line and bar of `part`'s label:
  // heavy for the leading part under Unicode, matching its heavy underline.
  // ASCII draws a heavy line as `^`, which belongs on the underline row alone,
  // so there the leading part's label line and bar are light like every other
  // label's.
  auto LabelWeight(const PreparedPart& part) const -> Terminal::LineWeight;

  // Draws `text` for `part` from (x, y), after its level word and in bold when
  // it leads, and returns the row after the text.
  auto DrawLabelText(const PreparedPart& part, llvm::StringRef text, int x,
                     int y, bool is_label_row) -> int;

  // Draws a label's bar in column `x` as a separate line on each of rows
  // `[y, next_y)` instead of one line through them, so that each cell is drawn
  // as a bar: the one the label line meets is a tee instead of the corner an
  // end would be, and a label of one row is framed the same as a label of ten.
  auto DrawBar(int x, int y, int next_y, const Style& style,
               Terminal::LineWeight weight) -> void;

  // Draws a label line from the underline row down to row `y` in column `x`,
  // ending in the upper half of the corner that turns toward the label. The
  // leading part's label row is the first under its underline, so its label
  // line runs down a clear column and joins what it meets. Every other label
  // line crosses the label rows drawn above it without joining them, so what
  // they drew stays whole, and the upper half of its corner is drawn
  // separately.
  auto DrawLabelLine(const PreparedPart& part, int x, int y, const Style& style,
                     Terminal::LineWeight weight) -> void;

  // Draws a label line from the underline row down to the row above `to_y` in
  // column `x`. It breaks around every line drawn earlier and skips the rows of
  // out-dented blocks, resuming below each one.
  auto DrawCrossing(int x, int to_y, const Style& style) -> void;

  Buffer* buffer_;
  Glyphs glyphs_;
  Charset charset_;
  int frame_x_;
  int content_x_;
  int columns_;

  // The frame's glyph on underline and label rows is in the frame's column, so
  // it takes one color for the whole diagnostic instead of the color of the
  // range on that row: the leading part's, which the `->` and the reported
  // line's number also take.
  Style dotted_frame_style_;

  // The row the underlines were drawn on, which every label below it connects
  // back up to. A label can be several rows down from it, since the labels on
  // one underline row stack.
  int underline_y_ = -1;

  // The rows below the underline row that out-dented labels' text occupies, as
  // `[first, past-the-last)` pairs in drawing order. A label line descending
  // past them skips these rows.
  llvm::SmallVector<std::pair<int, int>> outdented_rows_;
};

}  // namespace

auto FrameDrawer::Draw(llvm::ArrayRef<Row> rows) -> void {
  // A row that wraps pushes the ones below it down, so `y` follows what was
  // drawn instead of counting rows.
  int y = 0;
  for (const Row& row : rows) {
    int next_y = y + 1;
    switch (row.kind) {
      case Row::Kind::Location:
        next_y = DrawLocationRow(*row.part, y);
        break;
      case Row::Kind::Path:
        next_y = DrawPath(*row.part, y);
        break;
      case Row::Kind::Source:
        next_y = DrawSource(row, y);
        break;
      case Row::Kind::Underline:
        next_y = DrawUnderlineRow(row, y);
        break;
      case Row::Kind::Label:
        next_y = DrawLabel(row, y);
        break;
      case Row::Kind::RoutedLabel:
        next_y = DrawRoutedLabel(row, y);
        break;
      case Row::Kind::LocationLabel:
        next_y = DrawLocationLabel(row, y);
        break;
      case Row::Kind::Elision:
        break;
    }
    DrawFrame(row.kind, y, next_y);
    y = next_y;
  }

  // The closing rule turns the corner the frame's last row left for it.
  buffer_->DrawVerticalLine(frame_x_, y, 1, FrameStyle(), LineEnd::Edge,
                            LineEnd::Center);
  buffer_->DrawHorizontalLine(frame_x_, y, CloseRuleWidth, FrameStyle(),
                              LineEnd::Center, LineEnd::Edge);
}

// Draws `╭─┤ location`. The bracket is drawn as lines so that the character set
// decides how it looks, and so that a path row above can connect down into it.
auto FrameDrawer::DrawLocationRow(const PreparedPart& part, int y) -> int {
  buffer_->DrawHorizontalLine(frame_x_, y, 3, FrameStyle());
  buffer_->DrawVerticalLine(frame_x_ + 2, y, 1, FrameStyle(), LineEnd::Edge,
                            LineEnd::Edge);
  buffer_->DrawText(frame_x_ + LocationPrefixWidth, y,
                    part.location.empty() ? llvm::StringRef(NoLocationText)
                                          : llvm::StringRef(part.location),
                    Style());
  // The `->` points to where the leading part's text is placed. With no source
  // row for it, that is this location row.
  if (part.is_leading && !part.has_source) {
    buffer_->DrawText(0, y, LineArrow, part.style);
  }
  return y + 1;
}

// Draws a step of a path. Its line descends into the bracket before the file
// name in the location row below, so the row shows that it leads there. It is
// drawn downwards so that several steps stack into one line.
auto FrameDrawer::DrawPath(const PreparedPart& part, int y) -> int {
  int bracket_x = frame_x_ + 2;
  buffer_->DrawVerticalLine(bracket_x, y, 2, FrameStyle());
  buffer_->DrawHorizontalLine(bracket_x, y, LineToTextWidth - 1, FrameStyle(),
                              LineEnd::Center, LineEnd::Edge);
  std::string text = PathText(part);
  int text_x = bracket_x + LineToTextWidth;
  Buffer::DrawEnd end = buffer_->DrawText(text_x, y, text, PathStyle());
  buffer_->DrawText(end.x, end.y, part.kind_suffix, KindStyle());
  return y + 1;
}

auto FrameDrawer::DrawSource(const Row& row, int y) -> int {
  DrawLineNumber(*buffer_, frame_x_, y, row.line_number, row.points_here);
  buffer_->DrawText(content_x_, y, row.text, Style());
  return y + 1;
}

auto FrameDrawer::DrawUnderlineRow(const Row& row, int y) -> int {
  underline_y_ = y;
  outdented_rows_.clear();
  for (const PreparedPart* span : row.spans) {
    DrawUnderline(*buffer_, content_x_, y, *span, span->underline_style);
  }
  return y + 1;
}

// Draws a label to the right of its label line, which meets the bar framing it
// on its first row, where the text and any level word begin, so that following
// the line from the underline arrives at the start of the sentence. The bar
// continues down every row the text wraps onto.
auto FrameDrawer::DrawLabel(const Row& row, int y) -> int {
  const PreparedPart& part = *row.part;
  int line_x = content_x_ + part.source.line_column;
  int text_x = content_x_ + part.source.label_x;
  int bar_x = text_x - LabelFrameWidth;
  const Style& style = part.underline_style;
  Terminal::LineWeight weight = LabelWeight(part);
  int next_y = DrawLabelText(part, row.text, text_x, y, /*is_label_row=*/true);
  DrawBar(bar_x, y, next_y, style, weight);
  DrawLabelLine(part, line_x, y, style, weight);
  buffer_->DrawHorizontalLine(line_x, y, bar_x - line_x + 1, style,
                              LineEnd::Center, LineEnd::Center, weight);
  return next_y;
}

// Draws a label out-dented past its label line, where there is no room to meet
// it from the left. The bar framing it connects instead: its top turns right
// along the row under the underline row and runs to the label line. Meeting the
// label at its top lets the bar be directly against the text, which is the
// width out-denting is for.
auto FrameDrawer::DrawRoutedLabel(const Row& row, int y) -> int {
  const PreparedPart& part = *row.part;
  int line_x = content_x_ + part.source.line_column;
  int bar_x = content_x_;
  int text_x = bar_x + LabelFrameWidth;
  const Style& style = part.underline_style;
  Terminal::LineWeight weight = LabelWeight(part);
  DrawLabelLine(part, line_x, y, style, weight);
  buffer_->DrawHorizontalLine(bar_x, y, line_x - bar_x + 1, style,
                              LineEnd::Center, LineEnd::Center, weight);
  int next_y =
      DrawLabelText(part, row.text, text_x, y + 1, /*is_label_row=*/true);
  buffer_->DrawVerticalLine(bar_x, y, next_y - y, style, LineEnd::Center,
                            LineEnd::Edge, weight);
  // The block occupies its rows from the bar to the right edge, blank cells
  // included, so a later label line skips those rows instead of showing through
  // the gaps between words.
  outdented_rows_.push_back({y + 1, next_y});
  return next_y;
}

// Draws the text of a part with no source under its location row, the way a
// label is placed under its underline, with the label line's corner in the
// source's first column. There is no underline above it, so the line starts at
// the corner.
auto FrameDrawer::DrawLocationLabel(const Row& row, int y) -> int {
  const PreparedPart& part = *row.part;
  int line_x = content_x_;
  int text_x = content_x_ + LabelIndentWidth;
  int bar_x = text_x - LabelFrameWidth;
  const Style& style = part.underline_style;
  Terminal::LineWeight weight = LabelWeight(part);
  int next_y = DrawLabelText(part, row.text, text_x, y, /*is_label_row=*/false);
  DrawBar(bar_x, y, next_y, style, weight);
  buffer_->DrawVerticalLine(line_x, y, 1, style, LineEnd::Edge, LineEnd::Center,
                            weight);
  buffer_->DrawHorizontalLine(line_x, y, bar_x - line_x + 1, style,
                              LineEnd::Center, LineEnd::Center, weight);
  return next_y;
}

auto FrameDrawer::DrawFrame(Row::Kind kind, int y, int next_y) -> void {
  // The frame's line runs the whole height of the row, edge to edge, so that
  // consecutive rows' lines join without either reaching into the other's
  // cells. Only the first row's line starts at a center, which leaves that
  // cell a corner for the location row to turn. The line is drawn a row at a
  // time: drawing it first would need its full height, which isn't known until
  // the rows that wrap are drawn, and drawing it last would cover the glyphs
  // drawn over it below.
  buffer_->DrawVerticalLine(frame_x_, y, next_y - y, FrameStyle(),
                            y == 0 ? LineEnd::Center : LineEnd::Edge,
                            LineEnd::Edge);
  char32_t glyph = glyphs_.dotted_frame;
  Style style = dotted_frame_style_;
  switch (kind) {
    case Row::Kind::Underline:
    case Row::Kind::Label:
    case Row::Kind::RoutedLabel:
    case Row::Kind::LocationLabel:
      break;
    case Row::Kind::Elision:
      glyph = glyphs_.elision;
      style = FrameStyle();
      break;
    case Row::Kind::Location:
    case Row::Kind::Path:
    case Row::Kind::Source:
      return;
  }
  for (int row_y : llvm::seq(y, next_y)) {
    buffer_->DrawCodePoint(frame_x_, row_y, glyph, style);
  }
}

auto FrameDrawer::LabelWeight(const PreparedPart& part) const
    -> Terminal::LineWeight {
  return part.is_leading && charset_ == Charset::Utf8
             ? Terminal::LineWeight::Heavy
             : Terminal::LineWeight::Light;
}

auto FrameDrawer::DrawLabelText(const PreparedPart& part, llvm::StringRef text,
                                int x, int y, bool is_label_row) -> int {
  Style style;
  if (part.is_leading) {
    x = buffer_->DrawText(x, y, part.level_text, part.style).x;
    style = Style().Bold();
  }
  return DrawWrapped(*buffer_, x, y, columns_ - x, text,
                     part.KindSuffixOn(is_label_row), style);
}

auto FrameDrawer::DrawBar(int x, int y, int next_y, const Style& style,
                          Terminal::LineWeight weight) -> void {
  for (int bar_y : llvm::seq(y, next_y)) {
    buffer_->DrawVerticalLine(x, bar_y, 1, style, LineEnd::Edge, LineEnd::Edge,
                              weight);
  }
}

auto FrameDrawer::DrawLabelLine(const PreparedPart& part, int x, int y,
                                const Style& style, Terminal::LineWeight weight)
    -> void {
  CARBON_CHECK(underline_y_ >= 0,
               "A label row follows the underline row it is placed under.");
  if (part.is_leading) {
    DrawLeadingLabelLine(*buffer_, x, underline_y_, y, style, weight);
  } else {
    DrawCrossing(x, y, style);
    buffer_->DrawVerticalLine(x, y, 1, style, LineEnd::Edge, LineEnd::Center);
  }
}

auto FrameDrawer::DrawCrossing(int x, int to_y, const Style& style) -> void {
  int from_y = underline_y_;
  LineEnd start = LineEnd::Center;
  for (auto [top, bottom] : outdented_rows_) {
    if (bottom <= from_y || top >= to_y) {
      continue;
    }
    if (top > from_y) {
      buffer_->DrawCrossingVerticalLine(x, from_y, top - from_y, style, start);
    }
    from_y = bottom;
    start = LineEnd::Edge;
  }
  if (from_y < to_y) {
    buffer_->DrawCrossingVerticalLine(x, from_y, to_y - from_y, style, start);
  }
}

// Draws the framed form: one frame holding every location row, snippet, and
// label, with the leading part's text placed under the range that is wrong.
// `Render` chose the geometry; `RowLayout::AddLine` windows the spans, since it
// knows which of them share a view of a line.
auto Renderer::RenderFramed(Terminal::OutputBufferRef out,
                            llvm::MutableArrayRef<Internal::PreparedPart> parts,
                            int leading, Level level, int frame_x,
                            int content_x, int source_columns,
                            int columns) const -> void {
  Theme problem_theme = LevelTheme(level, capabilities_);
  Theme explanation_theme = ExplanationTheme(capabilities_);

  // Every layout decision is made against `columns`, so the buffer is built for
  // that rather than from the capabilities directly: where they held no width,
  // this is the one the layout chose.
  Terminal::Capabilities layout_capabilities = capabilities_;
  layout_capabilities.columns = columns;
  Buffer buffer(layout_capabilities);

  llvm::SmallVector<Row> rows =
      RowLayout(buffer, parts, leading, problem_theme, explanation_theme,
                source_columns, content_x, columns)
          .LayOut();

  FrameDrawer(buffer, capabilities_.charset, frame_x, content_x, columns,
              UnderlineStyle(parts[leading].style))
      .Draw(rows);
  buffer.Render(out, capabilities_.color_mode);
}

// Draws the row `part` takes in the compact form -- its location with the
// extent of its range, level word, words, and kind -- and returns whether one
// was drawn: a part that only underlines a range has nothing to say on a row of
// its own.
//
// Location information reads words first, leading to its own location, as it
// does in the frame. Reading it the other way around would make the words say
// the wrong thing.
static auto DrawCompactRow(Buffer& buffer, int y, const PreparedPart& part)
    -> bool {
  // With no frame for a label to hang off, its words serve as the part's own.
  const std::string& words = part.text.empty() ? part.label : part.text;
  if (part.role == Part::LocationInfo) {
    Buffer::DrawEnd end = buffer.DrawText(0, y, PathText(part), PathStyle());
    buffer.DrawText(end.x, end.y, part.kind_suffix, KindStyle());
    return true;
  }
  if (words.empty()) {
    return false;
  }
  // Each run picks up where the one before it ended rather than being placed
  // at a width measured separately, so the two can't disagree.
  Buffer::DrawEnd end = {.x = 0, .y = y};
  if (!part.location.empty()) {
    // The location carries the extent of the range as `-<end>`, which is the
    // one place this form has to say it; the framed form draws it instead.
    std::string location = part.location;
    if (part.location_end_column > 0) {
      location += "-" + std::to_string(part.location_end_column);
    }
    end = buffer.DrawText(0, y, location + ": ", Style().Bold());
  }
  end = buffer.DrawText(end.x, end.y, part.level_text, part.style);
  end = buffer.DrawText(end.x, end.y, words, Style());
  buffer.DrawText(end.x, end.y, part.kind_suffix, KindStyle());
  return true;
}

auto Renderer::RenderCompact(Terminal::OutputBufferRef out,
                             llvm::ArrayRef<Internal::PreparedPart> parts) const
    -> void {
  // Nothing here wraps, so the width binds nothing: each row runs as wide as
  // the runs drawn into it. The capabilities still decide how those are
  // measured and colored.
  Buffer buffer(capabilities_);
  int y = 0;
  for (const PreparedPart& part : parts) {
    if (DrawCompactRow(buffer, y, part)) {
      ++y;
    }
  }
  buffer.Render(out, capabilities_.color_mode);
}

auto FormatLocation(const Loc& loc) -> std::string {
  if (loc.filename.empty()) {
    return "";
  }
  // A filename is drawn on one row like anything else, so a control byte in it
  // -- a newline most of all -- would break the frame or split a compact line
  // in two. The bytes are the name of a real file, so they are replaced rather
  // than dropped: the location still reads, and nothing it holds can move the
  // cursor.
  std::string result;
  for (char c : loc.filename.take_front(MaxDrawnBytes)) {
    result += static_cast<unsigned char>(c) < 0x20 || c == 0x7f ? ' ' : c;
  }
  // The parts drop off the right as they become unknown.
  if (loc.line_number > 0) {
    result += ":" + std::to_string(loc.line_number);
    if (loc.column_number > 0) {
      result += ":" + std::to_string(loc.column_number);
    }
  }
  return result;
}

auto PrintSnippet(llvm::raw_ostream& out, const Loc& loc, int indent) -> void {
  // Crash output is plain: it is going into a stack trace, and whatever is
  // writing it has bigger problems than color.
  Metrics metrics(Charset::Ascii);
  llvm::SmallString<256> bytes;

  if (loc.column_number <= 0) {
    return;
  }
  Glyphs glyphs = Glyphs::For(Charset::Ascii);
  std::string number =
      loc.line_number > 0 ? std::to_string(loc.line_number) : std::string();
  int frame_x = indent + std::max<int>(number.size(), 1) + 1;
  int content_x = frame_x + FrameToContentWidth;
  PreparedSource source = PrepareSource(
      metrics, loc.line, Bytes(loc.column_number), Bytes(loc.length));
  WindowSource(metrics, source, FormattedSourceColumns, source.span_start);

  Buffer buffer(content_x + source.Width(metrics), Charset::Ascii);
  buffer.DrawVerticalLine(frame_x, /*y=*/0, /*length=*/2, FrameStyle());
  buffer.DrawText(frame_x - 1 - number.size(), 0, number, FrameStyle());
  buffer.DrawText(content_x, 0, source.text, Style());
  buffer.DrawCodePoint(frame_x, 1, glyphs.dotted_frame, FrameStyle());
  buffer.DrawHorizontalLine(content_x + source.span_start, 1,
                            source.span_length, Style(), LineEnd::Edge,
                            LineEnd::Edge);
  buffer.Render(bytes, Terminal::ColorMode::NoColor);
  out << bytes;
}

}  // namespace Carbon::Diagnostics

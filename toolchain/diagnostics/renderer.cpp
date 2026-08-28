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
// the stub next to it that a source row leaves blank.
static constexpr int FrameToContentWidth = 2;

// Points at the lines the problem is on -- every line holding a range that is
// part of it -- from the margin outside the line numbers. A merged snippet can
// open on some other line, so without this there is nothing saying which lines
// the problem is actually on.
//
// Two ASCII characters rather than one of the arrows, which all read better and
// none of which measure reliably: they are East Asian Ambiguous, so a font may
// draw one across two columns while the width tables call it one, and a
// terminal falling back to such a font pushes the whole row over or clips the
// glyph. This is two columns because it is two characters, so nothing can
// disagree about it, and a font with programming ligatures draws it as an arrow
// anyway.
static constexpr llvm::StringLiteral LineMark = "->";

// The columns the mark spends before the line numbers. No space after it: the
// numbers are right-aligned in a field of their own, so a number shorter than
// that field already leaves a gap, and a number filling it is the widest there
// is to keep the mark off.
static constexpr int MarginWidth = LineMark.size();

// The columns an anchor spends before the location it names: `╭─┤ `.
static constexpr int AnchorPrefixWidth = 4;

// The columns a connector spends before what it reaches: three of line, and
// then either a space or, where it reaches a label, the bar framing it.
static constexpr int ConnectorWidth = 4;

// The columns the bar framing a label's left side spends before its text:
// `┤ `. The bar runs down every row the label takes, so that what belongs to
// the label is bounded rather than left to be inferred from indentation.
static constexpr int LabelFrameWidth = 2;

// The columns between a label's anchor and its text: the connector reaching
// the bar that frames it, and the bar's own space before the words.
static constexpr int LabelIndentWidth = ConnectorWidth - 1 + LabelFrameWidth;

// The columns a rule bracketing a snippet reaches across, starting at the
// frame. Long enough to read as a divider rather than as another branch off the
// frame, which is what the connector on a context row looks like.
static constexpr int SnippetRuleWidth = 13;

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

// What marks each end of a source line that windowing elided.
static constexpr llvm::StringLiteral WindowMarker = "...";

// The fewest columns of source a window shows, however narrow the terminal
// is.
static constexpr int MinWindowColumns = 16;

// The most lines shown rather than elided between two spans in one file. One
// line of context is worth as much as the row saying it was skipped.
static constexpr int MaxShownGap = 1;

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

// Returns the style the marks under the source take, which is the color of
// whatever they belong to without its weight. Bold reaches the line-drawing
// glyphs unevenly -- several terminals leave them alone and others only
// brighten them -- so the weight says nothing reliable there while it still
// says something on words.
static auto MarkStyle(const Style& style) -> Style { return style.Bold(false); }

// The colors one theme is drawn in: what a diagnostic of that level, or the
// notes explaining one, take for their words and for the marks under the source
// that belong to them.
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
// these are drawn as marks a pixel tall.
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
// underline beneath the source that belongs to it, so that a mark and the
// sentence explaining it look like one thing.
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
static auto ContextStyle() -> Style { return Style().Dim(); }

namespace {

// The glyphs that vary with the character set.
//
// Everything else the frame is made of comes from `Terminal::Buffer`'s line
// drawing, which picks its own glyphs and forms the corners and tees where
// lines meet.
struct Glyphs {
  // Marks a row that annotates source rather than being source, so that which
  // rows are the reader's code is visible at a glance.
  char32_t annotation;
  // Marks where lines of source were skipped.
  char32_t elision;

  static auto For(Charset charset) -> Glyphs {
    if (charset == Charset::Utf8) {
      return {.annotation = U'·', .elision = U'┆'};
    }
    // An ASCII terminal has one broken bar to spend on both the annotation and
    // the elision. A row holding nothing but the bar is an elision; one with an
    // underline after it is not.
    return {.annotation = U':', .elision = U':'};
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
// `Terminal::Buffer` addresses its grid with, so what this marks is the count
// that is not a coordinate.
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

  // The column a label's connector runs down, which layout sets to whichever
  // end of the span the label attaches to. Until then it is the middle, which
  // is what a span with no label of its own is measured from.
  int anchor = 0;

  // Where the underline stops at each end. Layout sets an end to `Center` when
  // another range begins or ends in the next column, so that the two marks
  // don't run together; an end with nothing beside it covers its whole column.
  LineEnd left_end = LineEnd::Edge;
  LineEnd right_end = LineEnd::Edge;

  // Whether the label's connector is drawn in place of the underline rather
  // than hanging off it. A mark of one column with an end pulled in has no
  // junction to offer, and the line down its column says the column is marked
  // without saying the mark carries on anywhere.
  bool connector_stands_in = false;

  // Returns whether the underline covers `column` at all.
  auto Covers(int column) const -> bool {
    return column >= span_start && column < span_start + span_length;
  }

  // Returns whether the underline runs out through the left or the right side
  // of `column`. An end pulled in to clear the range beside it gives up the
  // side it was pulled in on, which is the side a connector could not tee into.
  auto ReachesLeftOf(int column) const -> bool {
    return Covers(column) && (column > span_start || left_end == LineEnd::Edge);
  }
  auto ReachesRightOf(int column) const -> bool {
    return Covers(column) && (column < span_start + span_length - 1 ||
                              right_end == LineEnd::Edge);
  }

  // Returns whether any column of the underline runs out through both of its
  // sides, which is what a connector needs one to do to tee into it.
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

  // Whether this part's range came from the message's own location rather than
  // from a range attached to the diagnostic. Only the message has one.
  //
  // Every diagnostic marks at least one range as part of the problem. Where one
  // was attached, it says where the problem is better than the message's
  // location does, and this range gives way to it: a range containing the
  // location hosts the message's words, and one that doesn't leaves the message
  // marking nothing of its own.
  bool from_message_loc = false;

  // The word leading this part's text, including its separator, and empty for
  // location information, which leads with its own words instead.
  llvm::StringRef level_text;

  // The style of that word and of this part's text.
  Style style;

  // The style of the underline under this part's source, and of the connector
  // and bar of the label hanging off it. The rotation moves this along the
  // theme's ramp, which is why it is not `style`: the level word, the mark
  // in the margin, and the line number beside the reported line all say what
  // the diagnostic is, so where a range sits on a line must not recolor them.
  Style mark_style;

  // `file:line:column`, with unknown parts left off, and empty for a part
  // with no location at all.
  std::string location;

  // The last byte column of the range starting at `location`, or 0 for a
  // range of one column or of unknown extent. The compact form appends it to
  // the location as `-<end>`, which is all it has to say the extent with; the
  // framed form says it with the underline instead.
  int32_t location_end_column = 0;

  // The part's words when they go on a row of their own: the headline, and any
  // part with no source to hang them against. Empty when they hang as a label.
  std::string text;

  // The part's words when they hang against the source it marks, which is any
  // part that has source and isn't the headline. Empty otherwise.
  std::string label;

  // ` [Kind]`, or empty when kinds aren't being included. It goes at the end
  // of the last row carrying the part's words, so that a test matching on it
  // always finds it last, and so that a part drawn on two rows doesn't get it
  // twice.
  std::string kind_suffix;

  // Returns the suffix for a row, which is empty unless that row is the last
  // one carrying this part's words.
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

// How a label's connector meets the underline it hangs from.
enum class Attachment : int8_t {
  // Tees into the underline where it hangs from. Line drawing forms the
  // junction the same way it forms the frame's own.
  Tee,
  // Stands in for the mark, in the one column it had, because the underline
  // offers no column to tee into: a mark that stops at a cell's center to
  // clear the range beside it would form a corner there, and a corner reads as
  // a mark turning rather than as one a connector leaves.
  Through,
};

// Where a label's connector meets its underline, and the column it runs down.
struct Placement {
  Attachment attachment;
  int anchor;
};

// A row of the rendering, laid out before anything is measured or drawn.
struct Row {
  enum class Kind : int8_t {
    // A message on a row of its own, inside the frame.
    Message,
    // The location the snippet below it comes from.
    Anchor,
    // A line of source, with its number beside the frame.
    Source,
    // The underline marking a span in the source above.
    Annotation,
    // The text hanging off that underline, framed by a bar the underline's
    // connector reaches.
    Label,
    // A label that wouldn't fit hanging off its span, out-dented to the column
    // the source starts in with its own frame reaching back to the span.
    RoutedLabel,
    // A stand-in for lines of source that were skipped.
    Elision,
    // Nothing but the frame, setting a snippet off from the anchor above it.
    Gap,
    // The rule closing off one snippet before the next begins.
    Separator,
    // How the location in the anchor below it was reached.
    Context,
  };

  Kind kind;
  // The part this row belongs to, or null for a row that draws only frame and
  // for a source line shown between two spans.
  const PreparedPart* part = nullptr;
  // `Annotation` rows: every span underlined, widest first, which is the order
  // they are drawn in so that the narrowest mark on a column is what shows.
  llvm::SmallVector<const PreparedPart*, 0> spans;
  // `Source` rows: the part whose style the margin mark and line number
  // take, set on every line holding a range that is part of the problem, and
  // null on every other line.
  const PreparedPart* points_here = nullptr;
  // `Source` rows: the number to draw, or -1 when there is none to show.
  int line_number = -1;
  // `Source` rows: the text.
  std::string text;
};

}  // namespace

// Returns whether `byte` continues a UTF-8 symbol that started before it.
static auto IsContinuation(char byte) -> bool {
  return (static_cast<unsigned char>(byte) & 0xc0) == 0x80;
}

// Returns `line` with tabs expanded and unrenderable bytes escaped, along with
// where the span of `span_bytes` bytes at `column` lands in the result.
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
  result.anchor = result.span_start + result.span_length / 2;
  return result;
}

// Narrows `source` to `columns` around `focus`, marking each end that was
// elided.
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
  // still windowed to bring that mark inside them.
  int width = source.Width(metrics);
  if (width <= columns) {
    return;
  }

  // Both markers are accounted for whether or not both are needed, so that how
  // much source a window shows doesn't depend on where in the line it lands.
  auto marker_width = static_cast<int>(WindowMarker.size());
  int inner = std::max(columns - 2 * marker_width, MinWindowColumns);
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
    text.append(WindowMarker.begin(), WindowMarker.end());
    shift = marker_width;
  }
  text.append(window.begin(), window.end());
  if (!rest.empty()) {
    text.append(WindowMarker.begin(), WindowMarker.end());
  }

  // Both ends of the span move by what was really dropped from in front of it,
  // and each is clamped into the window's own columns. Clamping the ends rather
  // than carrying the length is what keeps a span the window clipped short --
  // one windowed around another span on its row, off one edge -- from
  // underlining the source that took its place: what was clipped away is not
  // marked. Every column is drawable and none is past the last cell, so the
  // mark stays a mark and never reaches past the grid.
  int visible = metrics.Width(text);
  int offset = shift - dropped;
  int old_end = source.span_start + source.span_length;
  source.span_start =
      std::clamp(source.span_start + offset, 0, std::min(visible, columns - 1));
  int span_end = std::clamp(old_end + offset, source.span_start + 1, columns);
  source.span_length = span_end - source.span_start;
  source.anchor = source.span_start + source.span_length / 2;
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
  for ([[maybe_unused]] int step : llvm::seq(part.line_number, number)) {
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
// `is_headline` says that the words lead the diagnostic, so they go on the
// headline row rather than against the source they point at.
static auto PreparePart(Metrics metrics, Part role, bool from_message_loc,
                        llvm::StringRef level_text, Style style, const Loc& loc,
                        std::string text, llvm::StringRef name,
                        bool is_headline, bool include_kind) -> PreparedPart {
  // Text past any legitimate size is truncated rather than handed to the
  // buffer, whose own limit on a single draw is a CHECK; nothing a diagnostic
  // says comes near either bound.
  if (text.size() > MaxDrawnBytes) {
    text.resize(MaxDrawnBytes);
  }

  PreparedPart result = {
      .role = role,
      .from_message_loc = from_message_loc,
      .level_text = level_text,
      .style = style,
      .mark_style = MarkStyle(style),
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

  // Words that lead the diagnostic go on its headline, and words with no source
  // to point at have nowhere to go but a row of their own. Everything else is
  // read against the code it marks.
  if (is_headline || !result.has_source) {
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
  // would spend rows and a connector on drawing nothing. It becomes what a
  // label with no words is: a mark and nothing else.
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
static auto ContextText(const PreparedPart& part) -> std::string {
  if (part.location.empty()) {
    return part.text;
  }
  if (part.text.empty()) {
    return part.location;
  }
  return part.text + ": " + part.location;
}

// Returns what a part drawn on a row of its own says. Location information
// that never found an anchor to lead into says where it is here instead.
static auto MessageRowText(const PreparedPart& part) -> std::string {
  return part.role == Part::LocationInfo ? ContextText(part) : part.text;
}

// Returns where `label`'s connector meets the underline it hangs from, and the
// column it runs down from there.
//
// The connector tees into the middle of the underline, which reads as pointing
// into the range rather than as running off an end of it, and slides left when
// the words don't fit -- but only as far as it has to, and never past where the
// range starts. Moving the connector is what a label gives up before it starts
// wrapping.
//
// An end pulled in to clear the range beside it has no junction to offer
// either, and a connector meeting one would form a corner, which reads as the
// underline turning down rather than as a line leaving it. So the connector
// steps a column away from that range, where the marks run out through both
// sides again. Where no column of its range does -- a single column, or two
// with a range on either side -- it takes a column instead of teeing into one,
// and the underline gives that column up to it.
//
// Whether a column offers a junction is asked of every mark on the row rather
// than of `against` alone, since a cell holds all of them together: a range
// drawn over this one covers what its own end gave up, and a connector meeting
// that reads as the tee it is drawn as. Asking only `against` would leave a
// connector running through a column another mark had filled, crossing it.
//
// `against` is what the connector leaves, which is the label's own range except
// for the message's: where a range drawn on the same row contains the message's
// location, the words belong to that range.
static auto PlaceLabel(Metrics metrics, const PreparedPart& label,
                       const PreparedSource& against,
                       llvm::ArrayRef<PreparedPart*> spans, int content_x,
                       int columns) -> Placement {
  int width = metrics.Width(label.label);
  int middle = against.span_start + against.span_length / 2;
  int anchor = middle;
  if (content_x + anchor + LabelIndentWidth + width > columns) {
    anchor = std::clamp(columns - content_x - LabelIndentWidth - width,
                        against.span_start, anchor);
  }
  auto tees_at = [&](int column) {
    return against.Covers(column) &&
           llvm::any_of(spans,
                        [&](const PreparedPart* span) {
                          return span->source.ReachesLeftOf(column);
                        }) &&
           llvm::any_of(spans, [&](const PreparedPart* span) {
             return span->source.ReachesRightOf(column);
           });
  };
  if (tees_at(anchor)) {
    return {Attachment::Tee, anchor};
  }
  // Only an end fails to offer a junction, so the step is inward, away from the
  // range that pulled that end in.
  int step = anchor == against.span_start ? 1 : -1;
  if (tees_at(anchor + step)) {
    return {Attachment::Tee, anchor + step};
  }
  // Nothing to tee into, so every label on this range takes the same column,
  // which is the one the underline is measured from rather than wherever the
  // words happened to have to start.
  return {Attachment::Through, middle};
}

// Draws the mark under `span`, in source drawn from column `x`, on row `y`.
static auto DrawUnderline(Buffer& buffer, int x, int y,
                          const PreparedPart& span, const Style& style)
    -> void {
  // A connector with no column to tee into is drawn down the mark's own column
  // in place of it. Only a mark of one column is ever left with none, so there
  // is nothing of it to draw beside the line: an end gives up its half column
  // only where the range can still carry its label without it.
  if (span.source.connector_stands_in) {
    return;
  }
  buffer.DrawHorizontalLine(x + span.source.span_start, y,
                            span.source.span_length, style,
                            span.source.left_end, span.source.right_end);
}

// Returns the part whose range stands in for `message_part`'s own, or null when
// nothing does and the message marks where it points.
//
// A range attached as part of the problem says where the problem is in terms
// the reader can see, which the message's location -- a point, and often one
// chosen for an editor to put a cursor on rather than for anyone to look at --
// does not. So a range marking the line that location names stands in for it
// entirely: the message marks nothing of its own, and whatever words it has
// left once a context leads the diagnostic hang from that range instead.
//
// It has to be that line. A range elsewhere says nothing about the code the
// location names, and taking its place would point the message's words at code
// they aren't about while dropping the line they are.
//
// The range containing the location is the one that takes it, since that is the
// one drawn over the column the location names. Failing that it is the first
// attached to the line, which is the one whose author put it first.
//
// `candidates` are the ranges drawn alongside the message's, which is what a
// range has to be to take its place: one drawn under an anchor of its own is
// read against the code below that one instead.
static auto FindMessageHost(llvm::ArrayRef<PreparedPart*> candidates,
                            const PreparedPart& message_part) -> PreparedPart* {
  PreparedPart* host = nullptr;
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
    if (!host) {
      host = candidate;
    }
  }
  return host;
}

// Recolors the marks of `spans`, which share one line, along their themes'
// ramps. Each theme counts its own way through its colors, left to right, so
// they run in the order the code they mark is read.
static auto RotateMarkStyles(llvm::ArrayRef<PreparedPart*> spans,
                             Theme problem_theme, Theme explanation_theme)
    -> void {
  llvm::SmallVector<PreparedPart*> across(spans);
  llvm::stable_sort(across,
                    [](const PreparedPart* lhs, const PreparedPart* rhs) {
                      return lhs->source.span_start < rhs->source.span_start;
                    });
  int totals[2] = {0, 0};
  for (const PreparedPart* span : across) {
    ++totals[span->role == Part::Problem];
  }
  int counts[2] = {0, 0};
  for (PreparedPart* span : across) {
    bool is_problem = span->role == Part::Problem;
    int& count = counts[is_problem];
    span->mark_style = RotatedStyle(
        span->mark_style, is_problem ? problem_theme : explanation_theme,
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
// one end but not both, since a connector needs a column the mark runs out
// through on both sides and after the left end gave way only the right one is
// left. Left is considered first, so the range that begins at a boundary is
// always the one that gives way there, and every boundary ends up with a gap.
// One column can spare neither end, and the accepted answer there is the line
// the connector leaves or a point where it has no connector.
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

// Sets each label's anchor to where its connector runs down, asked of the
// underline it hangs from against everything else marked on the line.
//
// `message_part` and `message_host` are as in `add_line` below: the message's
// words hang from the host's range when one has taken its place.
static auto PlaceLabels(Metrics metrics, llvm::ArrayRef<PreparedPart*> labels,
                        const PreparedPart* message_part,
                        const PreparedPart* message_host,
                        llvm::ArrayRef<PreparedPart*> spans, int content_x,
                        int columns) -> void {
  for (PreparedPart* label : labels) {
    const PreparedSource& against = label == message_part && message_host
                                        ? message_host->source
                                        : label->source;
    Placement placement =
        PlaceLabel(metrics, *label, against, spans, content_x, columns);
    label->source.anchor = placement.anchor;
    // A connector drawn in place of a mark is recorded on the range it stands
    // in for, not on the label that asked for it -- and on every range drawn
    // over exactly the same column, since a second mark there would be a half
    // mark beside the connector rather than a stroke through it, which is the
    // one thing that would leave a corner in the cell.
    if (placement.attachment == Attachment::Through) {
      for (PreparedPart* span : spans) {
        if (span->source.span_start == against.span_start &&
            span->source.span_length == against.span_length) {
          span->source.connector_stands_in = true;
        }
      }
    }
  }
}

// Returns the rows a diagnostic's frame holds, in the order they are drawn.
//
// `source_columns` bounds every source row: the lines shown between two spans
// as well as the spans' own, which are windowed here so that spans sharing a
// row share one view of their line. `buffer` is the one the rows will be drawn
// into, and is only measured against here: how far text wraps is its answer to
// give.
static auto LayOutRows(const Buffer& buffer,
                       llvm::MutableArrayRef<PreparedPart> parts, int headline,
                       Theme problem_theme, Theme explanation_theme,
                       int source_columns, int content_x, int columns)
    -> llvm::SmallVector<Row> {
  Metrics metrics = buffer.metrics();
  llvm::SmallVector<Row> rows;
  // What the headline is drawn in, which is what the margin mark and the line
  // number beside it take, so that they read as part of the same sentence.
  const PreparedPart* headline_part = &parts[headline];
  llvm::SmallVector<bool> drawn(parts.size(), false);
  // Location information waiting for the anchor it describes.
  llvm::SmallVector<const PreparedPart*> context;

  // Adds the rows drawing the spans that fall on one source line: the line
  // itself, the underlines beneath it, and the labels hanging off those.
  //
  // `message_part` is the range standing in for the part's own location, and
  // `message_host` the attached range that has taken its place, in which case
  // the part underlines nothing of its own and its words hang from the host.
  auto add_line = [&](llvm::ArrayRef<PreparedPart*> line,
                      PreparedPart* message_part, PreparedPart* message_host) {
    // Spans sharing the row are windowed around the first of them: they share
    // the one source row that shows their line, so windowing each around its
    // own span would land their underlines in different views of it. Spans
    // drawn on different rows never share a window, wherever in the file they
    // are.
    int focus = line.front()->source.span_start;
    for (PreparedPart* member : line) {
      WindowSource(metrics, member->source, source_columns, focus);
    }

    // Everything marking the line goes on the one row below it, so that the
    // marks read as marking that line rather than as a stack of rows each of
    // which has to be traced back up to it. Ranges that overlap are drawn
    // widest first, so the narrowest mark covering a column is the one that
    // shows: a range containing another says less about the columns they share
    // than the range inside it does, and a mark that is part of the problem
    // says more than one that only explains it.
    //
    // Ranges shouldn't overlap in the first place -- one covering another means
    // two parts of the diagnostic marking the same code -- but they are
    // attached by layers that never see each other, so this repairs what it is
    // given rather than checking a rule nothing is in a position to enforce.
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

    RotateMarkStyles(spans, problem_theme, explanation_theme);
    // The part's connector and bar follow the range hosting its words, so
    // the rotation's choice for the host is copied after it is made.
    if (message_host && llvm::is_contained(spans, message_host)) {
      message_part->mark_style = message_host->mark_style;
    }

    // The mark in the margin picks out the lines the problem is on, which is
    // every line holding a range that is part of it.
    bool marks_problem = llvm::any_of(spans, [](const PreparedPart* span) {
      return span->role == Part::Problem;
    });
    rows.push_back({.kind = Row::Kind::Source,
                    .part = line.front(),
                    .points_here = marks_problem ? headline_part : nullptr,
                    .line_number = line.front()->line_number,
                    .text = line.front()->source.text});

    llvm::SmallVector<PreparedPart*> labels;
    for (PreparedPart* span : spans) {
      if (!span->label.empty()) {
        labels.push_back(span);
      }
      // A range standing in for the part's location hangs the part's own
      // words from its underline alongside whatever it has to say itself.
      if (span == message_host && !message_part->label.empty()) {
        labels.push_back(message_part);
      }
    }

    // The ranges a connector will hang from, which is what decides whether a
    // range can afford to give up half a column at an end.
    llvm::SmallPtrSet<const PreparedPart*, 4> carries_label;
    for (PreparedPart* label : labels) {
      carries_label.insert(label == message_part && message_host ? message_host
                                                                 : label);
    }

    SeparateTouchingEnds(spans, carries_label);
    PlaceLabels(metrics, labels, message_part, message_host, spans, content_x,
                columns);

    // Right to left, so that each label's connector reaches across the ones
    // already hanging rather than descending through them.
    llvm::stable_sort(labels,
                      [](const PreparedPart* lhs, const PreparedPart* rhs) {
                        return lhs->source.anchor > rhs->source.anchor;
                      });

    rows.push_back(
        {.kind = Row::Kind::Annotation, .spans = {spans.begin(), spans.end()}});

    for (PreparedPart* label : labels) {
      // A single unbreakable word can be wider than what is left to the right
      // of where the label hangs, and hanging it there would push it out of the
      // frame. Wrapping never breaks a word, so the widest one is the least
      // width the label can be drawn in. Below that the label is out-dented to
      // the column the source starts in, where the whole width is available,
      // and the line reaching back to the connector is what says which range it
      // belongs to.
      bool routed = content_x + label->source.anchor + LabelIndentWidth +
                        buffer.MeasureWrapWidth(label->label) >
                    columns;
      rows.push_back(
          {.kind = routed ? Row::Kind::RoutedLabel : Row::Kind::Label,
           .part = label,
           .text = label->label});
    }
  };

  // Adds `part`'s own text on a row of its own, which the drawing wraps into
  // the column after the level word.
  //
  // TODO: No toolchain diagnostic should reach this: the goal is every part
  // attached to a location that reaches source, with one genuinely about a
  // file as a whole anchored at the file's start or end and drawn in a form of
  // its own. That restructuring starts with the emitting diagnostics.
  auto add_message_rows = [&](const PreparedPart& part) {
    rows.push_back({.kind = Row::Kind::Message,
                    .part = &part,
                    .text = MessageRowText(part)});
  };

  // Whether the last row drawn was part of a snippet, and so wants closing off
  // before the next one starts.
  auto in_snippet = [&] {
    if (rows.empty()) {
      return false;
    }
    switch (rows.back().kind) {
      case Row::Kind::Source:
      case Row::Kind::Annotation:
      case Row::Kind::Label:
      case Row::Kind::RoutedLabel:
      case Row::Kind::Elision:
      case Row::Kind::Gap:
        return true;
      case Row::Kind::Message:
      case Row::Kind::Anchor:
      case Row::Kind::Separator:
      case Row::Kind::Context:
        return false;
    }
  };

  // Whether a part has to open a block of its own rather than being merged
  // into an earlier one: it has something to say on a row of its own, or there
  // is location information above it that needs its anchor to lead into.
  llvm::SmallVector<bool> opens_block(parts.size(), false);
  bool context_pending = false;
  for (auto [index, part] : llvm::enumerate(parts)) {
    if (part.role == Part::LocationInfo) {
      context_pending = true;
      continue;
    }
    opens_block[index] =
        context_pending ||
        (static_cast<int>(index) != headline && !part.text.empty());
    context_pending = false;
  }

  for (auto [index, part] : llvm::enumerate(parts)) {
    if (drawn[index]) {
      continue;
    }
    drawn[index] = true;

    // Location information says how the location after it was reached, so it
    // waits and is then drawn above that location's anchor.
    if (part.role == Part::LocationInfo) {
      context.push_back(&part);
      continue;
    }

    if (in_snippet()) {
      rows.push_back({.kind = Row::Kind::Separator});
    }

    if (static_cast<int>(index) != headline && !part.text.empty()) {
      add_message_rows(part);
    }
    // An anchor is drawn where a snippet follows it, so a part with no source
    // to show gets none: its words go on a row of their own, and an anchor
    // above them would lead nowhere. The headline is the exception -- its
    // location is the diagnostic's, worth saying even when it names only a
    // file, as for a diagnostic about a file rather than about code in one.
    bool anchored = !part.location.empty() &&
                    (part.has_source || static_cast<int>(index) == headline);
    // Without an anchor below them there is nothing for the steps to lead into,
    // so they say where they are on rows of their own instead.
    for (const PreparedPart* reached_from : context) {
      if (anchored) {
        rows.push_back({.kind = Row::Kind::Context, .part = reached_from});
      } else {
        add_message_rows(*reached_from);
      }
    }
    context.clear();

    // The spans shown under one anchor: every later span in the same file,
    // shown as one view of it in source order with the lines between them
    // elided.
    llvm::SmallVector<PreparedPart*> group;
    if (part.has_source) {
      group.push_back(&part);
      for (size_t other : llvm::seq(index + 1, parts.size())) {
        if (!drawn[other] && !opens_block[other] && parts[other].has_source &&
            parts[other].file == part.file) {
          drawn[other] = true;
          group.push_back(&parts[other]);
        }
      }
      llvm::stable_sort(group,
                        [](const PreparedPart* lhs, const PreparedPart* rhs) {
                          return lhs->line_number < rhs->line_number;
                        });
    }

    if (anchored) {
      rows.push_back({.kind = Row::Kind::Anchor, .part = &part});
    }

    // The spans shown under one anchor are the ones drawn alongside each other
    // here, so the group is what a range has to be in to stand in for the
    // part's location: a range under an anchor of its own is read against
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
      // The part marks nothing of its own once something else has taken its
      // place, so the line it named is not part of the snippet either -- unless
      // an attached range is on it too. Leaving it in shows a line of source
      // with nothing marking it, which reads as a line the reader is meant to
      // find something in.
      llvm::erase(group, message_part);
    }

    // A snippet always starts a row below its anchor, so that the code doesn't
    // read as running on from the file name. Where the file goes on above the
    // first line shown, that row says so instead.
    if (!group.empty()) {
      int first_line = group.front()->line_number;
      rows.push_back(
          {.kind = first_line > 1 ? Row::Kind::Elision : Row::Kind::Gap});
    }

    // Spans on one line are drawn together, against the one row that shows it.
    const PreparedPart* previous = nullptr;
    for (size_t begin = 0; begin < group.size();) {
      size_t end = begin + 1;
      while (end < group.size() &&
             group[end]->line_number == group[begin]->line_number) {
        ++end;
      }
      if (previous) {
        int gap = group[begin]->line_number - previous->line_number - 1;
        if (gap > MaxShownGap) {
          rows.push_back({.kind = Row::Kind::Elision});
        } else if (gap > 0) {
          for (int number : llvm::seq(previous->line_number + 1,
                                      group[begin]->line_number)) {
            llvm::StringRef line = LineFromFile(*previous, number);
            if (line.data() == nullptr) {
              // Without the file there is no way to show what was skipped, and
              // a blank row would claim the line is empty.
              rows.push_back({.kind = Row::Kind::Elision});
              continue;
            }
            PreparedSource shown =
                PrepareSource(metrics, line, Bytes(1), Bytes(1));
            WindowSource(metrics, shown, source_columns, /*focus=*/0);
            rows.push_back({.kind = Row::Kind::Source,
                            .line_number = number,
                            .text = std::move(shown.text)});
          }
        }
      }
      add_line(llvm::ArrayRef(group).slice(begin, end - begin), message_part,
               message_host);
      previous = group[end - 1];
      begin = end;
    }
  }
  for (const PreparedPart* reached_from : context) {
    add_message_rows(*reached_from);
  }
  return rows;
}

// Draws the line running from an underline at `annotation_y` down to `meet_y`,
// in the column a label hangs from.
//
// It stops at the center of the cell the underline drew rather than redrawing
// it, so where the underline runs out through both sides of that cell the two
// form a junction. Where it stands in for a mark of one column instead,
// stopping at the center is what keeps the mark to the lower half of its cell,
// so that two of them in neighboring columns read as two marks rather than as
// one line drawn down both.
static auto DrawConnector(Buffer& buffer, int x, int annotation_y, int meet_y,
                          const Style& style) -> void {
  buffer.DrawVerticalLine(x, annotation_y, meet_y - annotation_y + 1, style,
                          LineEnd::Center, LineEnd::Center);
}

// Draws `number` in the columns before the frame, or nothing when there is no
// line to name.
//
// `points_here` is the message whose location is on this line, which takes the
// margin outside the numbers and colors the number to match what the headline
// says. Everything else keeps the frame's own color, so the one line the
// message is about is the one that isn't part of the frame.
static auto DrawLineNumber(Buffer& buffer, int frame_x, int y, int number,
                           const PreparedPart* points_here) -> void {
  if (number <= 0) {
    return;
  }
  std::string text = std::to_string(number);
  // The line the problem is on takes the headline's color and its weight, and
  // every other line number is left as light as the frame it belongs to, so
  // that the one being reported is the one that stands out rather than every
  // number competing with it.
  Style style = points_here ? points_here->style : FrameStyle().Bold(false);
  buffer.DrawText(frame_x - 1 - text.size(), y, text, style);
  if (points_here) {
    buffer.DrawText(0, y, LineMark, style);
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
                                 (llvm::Twine(text) + suffix).str())
             .y == y;
}

// Returns the row after what `DrawWrapped` below draws, without drawing it.
static auto MeasureWrapped(const Buffer& buffer, int x, int y, int columns,
                           llvm::StringRef text, llvm::StringRef suffix)
    -> int {
  Buffer::DrawEnd end =
      buffer.MeasureWrappedText(x, y, /*margin=*/x, columns, text);
  if (suffix.empty() || WrappedSharesRow(buffer, x, y, columns, text, suffix)) {
    return end.y + 1;
  }
  return end.y + 2;
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
    return buffer.DrawWrappedText(x, y, /*margin=*/x, columns, text, style).y +
           1;
  }
  Buffer::DrawEnd end =
      buffer.DrawWrappedText(x, y, /*margin=*/x, columns, text, style);
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
  int headline = -1;

  // Which colors this terminal gets, decided once: everything drawn takes one
  // of them.
  Theme problem_theme = LevelTheme(diagnostic.level, capabilities_);
  Theme explanation_theme = ExplanationTheme(capabilities_);

  // What leads the diagnostic is drawn at its level and everything else as
  // something explaining it, except that a primary range is part of the problem
  // and so keeps the level's color wherever down the frame it sits.
  auto add_part = [&](bool is_headline, bool is_problem, bool from_message_loc,
                      const Loc& loc, std::string text, llvm::StringRef name) {
    if (is_headline) {
      headline = parts.size();
    }
    parts.push_back(
        PreparePart(metrics, is_problem ? Part::Problem : Part::Explanation,
                    from_message_loc,
                    is_headline ? LevelText(diagnostic.level) : ExplanationText,
                    ThemeStyle(is_problem ? problem_theme : explanation_theme),
                    loc, std::move(text), name, is_headline, include_kind_));
  };

  // The same file reached the same way is a path drawn once. A path says how a
  // file was got to, so every range marked in that file has the same one, and
  // repeating it under each of them says nothing the first didn't.
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
          /*level_text=*/"", ContextStyle(), step.loc, step.Format(), step.name,
          /*is_headline=*/false, include_kind_));
    }
  };

  // The contexts, then the message, then everything attached to explain it,
  // which is the order they are read in.
  for (const Context& context : diagnostic.contexts) {
    add_location_info(context.location_info, context.loc);
    // Only the leading context is the failure being reported; the rest name
    // operations it happened inside, which explain it.
    bool is_headline = &context == leading_context;
    add_part(is_headline, is_headline, /*from_message_loc=*/false, context.loc,
             context.Format(), context.name);
  }
  const Message& message = diagnostic.message;
  add_location_info(message.location_info, message.loc);
  // The message states the problem whether or not a context leads it: a
  // context gives it a sentence to sit under, and doesn't demote what it says
  // to an explanation of something else. Marking it otherwise puts an
  // informational connector on a range that is marked as part of the problem.
  bool message_leads = leading_context == nullptr;
  add_part(message_leads, /*is_problem=*/true, /*from_message_loc=*/true,
           message.loc, message.Format(), message.kind.name());
  for (const Label& label : diagnostic.labels) {
    add_location_info(label.location_info, label.loc);
    add_part(/*is_headline=*/false, label.category == LabelCategory::Primary,
             /*from_message_loc=*/false, label.loc, label.Format(), label.name);
  }
  CARBON_CHECK(headline >= 0, "Either the context or the message leads.");

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

  RenderFramed(out, parts, headline, diagnostic.level, frame_x, content_x,
               source_columns, columns);
}

// Draws the framed form: the headline, and under it one frame holding every
// anchor, snippet, and label. The geometry arrives from `Render`, which chose
// it; the spans are windowed in `LayOutRows`, which knows which of them share
// a view of a line.
auto Renderer::RenderFramed(Terminal::OutputBufferRef out,
                            llvm::MutableArrayRef<Internal::PreparedPart> parts,
                            int headline, Level level, int frame_x,
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
      LayOutRows(buffer, parts, headline, problem_theme, explanation_theme,
                 source_columns, content_x, columns);

  Glyphs glyphs = Glyphs::For(capabilities_.charset);

  // The headline, wrapped under itself so that a continuation of it can't read
  // as the start of another diagnostic.
  const PreparedPart& head = parts[headline];
  // The separator beside an annotation runs down the frame's own column and
  // reads as part of it, so it takes one color for the whole diagnostic rather
  // than the color of whichever range happens to be on that row. That color is
  // the headline's, which is what the mark in the margin and the line number
  // beside the reported line take.
  const Style annotation_style = MarkStyle(head.style);
  Buffer::DrawEnd end = buffer.DrawText(0, 0, head.level_text, head.style);
  int head_x = end.x;
  int head_bottom = DrawWrapped(buffer, head_x, 0, columns - head_x, head.text,
                                head.kind_suffix, Style().Bold());

  // Then everything else, hanging off one vertical line down the frame's
  // column. Its corners and tees come out of the junctions the buffer forms
  // where the stubs meet it, and rows that mark themselves draw their glyph
  // over it. A row that wraps pushes the ones below it down, so the cursor
  // follows what was drawn rather than counting rows.
  int frame_top = head_bottom;
  int y = frame_top;
  // The row the underlines were drawn on, which is where every label below it
  // reaches back up to. A label can be several rows down from it, since the
  // labels on one underline stack.
  int annotation_y = -1;
  // A routed label's words, held back until every connector is drawn; see the
  // `RoutedLabel` case below.
  struct DeferredText {
    int x;
    int y;
    int columns;
    const Row* row;
  };
  llvm::SmallVector<DeferredText> deferred_text;
  for (const Row& row : rows) {
    CARBON_CHECK(annotation_y >= 0 || (row.kind != Row::Kind::Label &&
                                       row.kind != Row::Kind::RoutedLabel),
                 "A label hangs off the underline laid out above it.");
    int next_y = y + 1;
    const PreparedPart* part = row.part;
    // What this row draws over the frame, from `mark_from` down, or nothing to
    // leave the frame's own line showing through.
    std::optional<char32_t> mark;
    Style mark_style = FrameStyle();
    int mark_from = y;
    switch (row.kind) {
      case Row::Kind::Message: {
        buffer.DrawHorizontalLine(frame_x, y, 2, FrameStyle(), LineEnd::Center,
                                  LineEnd::Edge);
        Buffer::DrawEnd end =
            buffer.DrawText(content_x + 1, y, part->level_text, part->style);
        next_y = DrawWrapped(
            buffer, end.x, y, columns - end.x, row.text,
            part->KindSuffixOn(/*is_label_row=*/false),
            part->role == Part::LocationInfo ? ContextStyle() : Style());
        mark = glyphs.annotation;
        mark_from = y + 1;
        break;
      }
      case Row::Kind::Anchor:
        // `╭─┤ location`. The bracket is drawn as lines so that the character
        // set decides how it looks, and so that the context above can reach
        // down into it.
        buffer.DrawHorizontalLine(frame_x, y, 3, FrameStyle());
        buffer.DrawVerticalLine(frame_x + 2, y, 1, FrameStyle(), LineEnd::Edge,
                                LineEnd::Edge);
        buffer.DrawText(frame_x + AnchorPrefixWidth, y, part->location,
                        Style());
        break;
      case Row::Kind::Context: {
        // The connector descends into the bracket before the file name in the
        // anchor below, so the row reads as leading to it. Drawn downwards so
        // that several of these stack into one line.
        int bracket_x = frame_x + 2;
        buffer.DrawVerticalLine(bracket_x, y, 2, FrameStyle());
        buffer.DrawHorizontalLine(bracket_x, y, ConnectorWidth - 1,
                                  FrameStyle(), LineEnd::Center, LineEnd::Edge);
        std::string text = ContextText(*part);
        int text_x = bracket_x + ConnectorWidth;
        Buffer::DrawEnd end = buffer.DrawText(text_x, y, text, ContextStyle());
        buffer.DrawText(end.x, end.y, part->kind_suffix, KindStyle());
        break;
      }
      case Row::Kind::Source:
        DrawLineNumber(buffer, frame_x, y, row.line_number, row.points_here);
        buffer.DrawText(content_x, y, row.text, Style());
        break;
      case Row::Kind::Annotation: {
        annotation_y = y;
        for (const PreparedPart* span : row.spans) {
          DrawUnderline(buffer, content_x, y, *span, span->mark_style);
        }
        mark = glyphs.annotation;
        mark_style = annotation_style;
        break;
      }
      case Row::Kind::Label: {
        // The label hangs to the right of the connector, which comes in from
        // the left to meet the bar framing it, at its middle row -- the row
        // above the middle when it has an even number of them.
        int anchor_x = content_x + part->source.anchor;
        int text_x = anchor_x + LabelIndentWidth;
        int bar_x = text_x - LabelFrameWidth;
        const Style& style = part->mark_style;
        next_y = DrawWrapped(buffer, text_x, y, columns - text_x, row.text,
                             part->KindSuffixOn(/*is_label_row=*/true));

        // The bar is a stroke on each of the label's rows rather than one line
        // running between them, so every cell of it reads as a bar: the one the
        // connector arrives at is a tee rather than the corner an end would be,
        // and a label of one row is framed the same as a label of ten.
        for (int bar_y : llvm::seq(y, next_y)) {
          buffer.DrawVerticalLine(bar_x, bar_y, 1, style, LineEnd::Edge,
                                  LineEnd::Edge);
        }
        // The connector runs from the underline down past whatever labels are
        // already hanging to the right of this one, all of which leave the
        // column it descends in clear -- except a routed label, whose words
        // are drawn last so that they win the cell; see below.
        int meet_y = y + (next_y - y - 1) / 2;
        DrawConnector(buffer, anchor_x, annotation_y, meet_y, style);
        buffer.DrawHorizontalLine(anchor_x, meet_y, ConnectorWidth, style);
        mark = glyphs.annotation;
        mark_style = annotation_style;
        break;
      }
      case Row::Kind::RoutedLabel: {
        // The label is out-dented past its anchor, so there is no room to reach
        // it from the left and the bar framing it is what reaches back instead:
        // its top turns right along the row under the underline and runs to the
        // anchor. Meeting the label at its top rather than its middle is what
        // lets the bar sit directly against the text, which is where the width
        // this form exists to find comes from.
        int anchor_x = content_x + part->source.anchor;
        int bar_x = content_x;
        int text_x = bar_x + LabelFrameWidth;
        const Style& style = part->mark_style;
        DrawConnector(buffer, anchor_x, annotation_y, y, style);
        buffer.DrawHorizontalLine(bar_x, y, anchor_x - bar_x + 1, style);

        // The words reach across the full width, so a label drawn after this
        // one -- to its left, and so below it -- runs its connector through
        // their rows. The words win the cells they share: they are measured
        // now, so the rows below land where they will, and drawn once every
        // connector is down, so a connector crossing this label is interrupted
        // by its words rather than striking through them.
        next_y =
            MeasureWrapped(buffer, text_x, y + 1, columns - text_x, row.text,
                           part->KindSuffixOn(/*is_label_row=*/true));
        deferred_text.push_back({.x = text_x,
                                 .y = y + 1,
                                 .columns = columns - text_x,
                                 .row = &row});
        buffer.DrawVerticalLine(bar_x, y, next_y - y, style, LineEnd::Center,
                                LineEnd::Edge);
        mark = glyphs.annotation;
        mark_style = annotation_style;
        break;
      }
      case Row::Kind::Elision:
        mark = glyphs.elision;
        break;
      case Row::Kind::Gap:
        // The frame's own line running through is all this row is.
        break;
      case Row::Kind::Separator:
        // The same rule that closes the diagnostic, tee'd because the frame
        // carries on past it.
        buffer.DrawHorizontalLine(frame_x, y, SnippetRuleWidth, FrameStyle(),
                                  LineEnd::Center, LineEnd::Edge);
        break;
    }

    // The frame's line runs the whole height of the row, edge to edge, so that
    // consecutive rows carry on into each other without either reaching into
    // the other's cells. Only the topmost row starts at a center, which is what
    // leaves that cell a corner for the anchor to turn. It can't be drawn in
    // one pass afterwards because a row that wraps pushes the ones below it
    // down, so how far it reaches isn't known until every row is drawn -- and
    // it can't be drawn in one pass beforehand because it would cover the
    // glyphs.
    buffer.DrawVerticalLine(frame_x, y, next_y - y, FrameStyle(),
                            y == frame_top ? LineEnd::Center : LineEnd::Edge,
                            LineEnd::Edge);
    if (mark) {
      for (int row_y : llvm::seq(mark_from, next_y)) {
        buffer.DrawCodePoint(frame_x, row_y, *mark, mark_style);
      }
    }
    y = next_y;
  }

  // The routed labels' words, drawn over whatever connectors crossed them.
  for (const DeferredText& text : deferred_text) {
    DrawWrapped(buffer, text.x, text.y, text.columns, text.row->text,
                text.row->part->KindSuffixOn(/*is_label_row=*/true));
  }

  // The closing rule turns the corner the frame's last row left for it.
  buffer.DrawVerticalLine(frame_x, y, 1, FrameStyle(), LineEnd::Edge,
                          LineEnd::Center);
  buffer.DrawHorizontalLine(frame_x, y, SnippetRuleWidth, FrameStyle(),
                            LineEnd::Center, LineEnd::Edge);
  buffer.Render(out, capabilities_.color_mode);
}

// Draws the row `part` takes in the compact form -- its location with the
// extent of its range, level word, words, and kind -- and returns whether one
// was drawn: a part that only marks a range has nothing to say on a row of its
// own.
//
// Location information reads words first, leading to its own location, as it
// does in the frame. Reading it the other way around would make the words say
// the wrong thing.
static auto DrawMessageRow(Buffer& buffer, int y, const PreparedPart& part)
    -> bool {
  // With no frame for a label to hang off, its words serve as the part's own.
  const std::string& words = part.text.empty() ? part.label : part.text;
  if (part.role == Part::LocationInfo) {
    Buffer::DrawEnd end =
        buffer.DrawText(0, y, ContextText(part), ContextStyle());
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
    if (DrawMessageRow(buffer, y, part)) {
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
  buffer.DrawCodePoint(frame_x, 1, glyphs.annotation, FrameStyle());
  buffer.DrawHorizontalLine(content_x + source.span_start, 1,
                            source.span_length, Style(), LineEnd::Edge,
                            LineEnd::Edge);
  buffer.Render(bytes, Terminal::ColorMode::NoColor);
  out << bytes;
}

}  // namespace Carbon::Diagnostics

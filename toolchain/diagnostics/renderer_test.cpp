// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/renderer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <string>

#include "common/raw_string_ostream.h"
#include "common/terminal/capabilities.h"
#include "common/terminal/metrics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"

namespace Carbon::Diagnostics {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;

// Returns a function that formats as `text`, whatever it is handed.
static auto Says(llvm::StringRef text) -> FormatFn {
  return
      [text = text.str()](llvm::StringLiteral /*format*/,
                          llvm::ArrayRef<llvm::Any> /*args*/) { return text; };
}

// Builds a step in the path by which a location was reached.
static auto MakeOrigin(Loc loc, llvm::StringLiteral text) -> LocationInfo {
  return {.loc = loc, .name = "TestOrigin", .format = text};
}

// Builds a message whose formatted text is `text`, reached by `origin`.
static auto MakeMessage(Loc loc, llvm::StringRef text,
                        llvm::SmallVector<LocationInfo, 0> origin = {},
                        Level level = Level::Error) -> Message {
  return {.kind = Kind::TestDiagnostic,
          .level = level,
          .loc = loc,
          .location_info = std::move(origin),
          .format = "",
          .format_args = {},
          .format_fn = Says(text)};
}

// Builds a label attached to `loc` and saying `text`, reached by `origin`. An
// empty `text` underlines the range and says nothing.
static auto MakeLabel(LabelCategory category, Loc loc, llvm::StringRef text,
                      llvm::SmallVector<LocationInfo, 0> origin = {}) -> Label {
  return {.category = category,
          .loc = loc,
          .location_info = std::move(origin),
          .name = "TestLabel",
          .format = "",
          .format_args = {},
          .format_fn = text.empty() ? FormatFn() : Says(text)};
}

// Builds a context saying `text` about `loc`, reached by `origin`.
static auto MakeContext(Loc loc, llvm::StringRef text,
                        llvm::SmallVector<LocationInfo, 0> origin = {})
    -> Context {
  return {.loc = loc,
          .location_info = std::move(origin),
          .name = "TestContext",
          .format = "",
          .format_args = {},
          .format_fn = Says(text)};
}

// The columns of source a diagnostic targets with no terminal to measure, which
// is `Renderer`'s own `FormattedSourceColumns`. Spelled again here rather than
// exposed: a test that pins the width should say which width it pins.
static constexpr int TargetSourceColumns = 80;

// The capabilities of a stream with no terminal behind it, which is what tests
// and redirected output get.
static auto Plain() -> Terminal::Capabilities { return {}; }

// The capabilities of a terminal too narrow to hold a frame, which is how a
// diagnostic with source to show still reaches the compact form.
static auto Narrow() -> Terminal::Capabilities {
  return {.is_terminal = true, .columns = 40};
}

// Renders `diagnostic` for `capabilities`.
static auto Render(const Terminal::Capabilities& capabilities,
                   const Diagnostic& diagnostic, bool snippets = true)
    -> std::string {
  Renderer renderer(capabilities);
  renderer.set_snippets(snippets);
  llvm::SmallString<256> bytes;
  renderer.Render(bytes, diagnostic);
  return std::string(bytes);
}

// Renders a diagnostic with a single message at `level`.
static auto RenderOne(const Terminal::Capabilities& capabilities, Level level,
                      Loc loc, llvm::StringRef text) -> std::string {
  return Render(capabilities,
                {.level = level, .message = MakeMessage(loc, text, {}, level)});
}

// The file the tests point spans into.
static constexpr llvm::StringLiteral File =
    "fn Run0() {}\n"
    "\n"
    "fn Main() {\n"
    "  Run0(1);\n"
    "}\n";

// Returns a location naming only a file, which is what a part attached at no
// particular code converts to.
static auto FileOnly() -> Loc { return {.filename = "foo.carbon"}; }

// Returns a location in `File` naming a line but no column. No producer emits
// one -- a location either reaches source or names at most a file -- but a
// `Loc` is a plain aggregate, so the renderer still has to draw it.
static auto LineOnly(int line_number) -> Loc {
  return {.filename = "foo.carbon", .line_number = line_number};
}

// Returns a location in `File`.
static auto At(int line_number, int column, int length) -> Loc {
  llvm::StringRef rest = File;
  for (auto _ : llvm::seq(1, line_number)) {
    rest = rest.split('\n').second;
  }
  return {.filename = "foo.carbon",
          .line = rest.split('\n').first,
          .file_text = File,
          .line_number = line_number,
          .column_number = column,
          .length = length};
}

TEST(RendererTest, ErrorWithSnippet) {
  // The common case: the location row starts the frame, the message's text is
  // placed under its own range after the level word, and the close ends it.
  EXPECT_THAT(RenderOne(Plain(), Level::Error, At(4, 3, 7), "bad call"),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: bad call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ASpanStartingWhereTheLineEndsIsStillDrawn) {
  // A location can name the column after the last character on a line: a
  // fix-it inserting there, or a token synthesized past the end of the line by
  // error recovery. The underline reaches one column further than the source
  // rather than the renderer refusing to draw it.
  Loc loc = {.filename = "foo.carbon",
             .line = "var x: i32",
             .file_text = "var x: i32",
             .line_number = 1,
             .column_number = 11,
             .length = 1};
  // Drawn as UTF-8 because that is where the span is walked back to the start
  // of whatever symbol it lands in, and there is no symbol at the end of a line
  // to walk back through.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  EXPECT_THAT(RenderOne(capabilities, Level::Error, loc, "missing `;`"),
              Eq("       ╭─┤ foo.carbon:1:11\n"
                 "->   1 │ var x: i32\n"
                 "       ·           ┳\n"
                 "       ·           ┗━━┫ error: missing `;`\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, WarningLevelWord) {
  EXPECT_THAT(RenderOne(Plain(), Level::Warning, At(1, 4, 4), "unused"),
              Eq("       .-| foo.carbon:1:4\n"
                 "->   1 | fn Run0() {}\n"
                 "       :    ^^^^\n"
                 "       :      '--| warning: unused\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, FrameMovesRightForLongLineNumbers) {
  Loc loc = {.filename = "foo.carbon",
             .line = "var x: i32;",
             .line_number = 123456,
             .column_number = 1,
             .length = 3};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("         .-| foo.carbon:123456:1\n"
                 "->123456 | var x: i32;\n"
                 "         : ^^^\n"
                 "         :  '--| error: bad\n"
                 "         '----\n"
                 ""));
}

TEST(RendererTest, NoLocationIsOneCompactRow) {
  EXPECT_THAT(RenderOne(Plain(), Level::Error, {}, "no file here"),
              Eq("error: no file here\n"));
}

TEST(RendererTest, NothingToFrameFallsBackToCompact) {
  EXPECT_THAT(
      RenderOne(Plain(), Level::Error, {.filename = "foo.carbon"}, "broken"),
      Eq("foo.carbon: error: broken\n"));
  Loc line_only = {.filename = "foo.carbon", .line_number = 3};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, line_only, "broken"),
              Eq("foo.carbon:3: error: broken\n"));
}

TEST(RendererTest, TabsExpandToStops) {
  Loc loc = {.filename = "foo.carbon",
             .line = "\tvar\tx;",
             .line_number = 1,
             .column_number = 6,
             .length = 1};
  // The tabs reach columns 8 and 16, so `x` lands in column 17.
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("       .-| foo.carbon:1:6\n"
                 "->   1 |         var     x;\n"
                 "       :                 ^\n"
                 "       :                 '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, UnprintableBytesAreEscaped) {
  Loc loc = {.filename = "foo.carbon",
             .line = "a\x01=b",
             .line_number = 1,
             .column_number = 4,
             .length = 1};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("       .-| foo.carbon:1:4\n"
                 "->   1 | a<01>=b\n"
                 "       :       ^\n"
                 "       :       '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, NonAsciiIsEscapedUnderAscii) {
  Loc loc = {.filename = "foo.carbon",
             .line = "x = \xc3\xa9;",
             .line_number = 1,
             .column_number = 1,
             .length = 1};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("       .-| foo.carbon:1:1\n"
                 "->   1 | x = <C3><A9>;\n"
                 "       : ^\n"
                 "       : '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, WideCharactersTakeTwoColumnsUnderUtf8) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  // `世界` is at byte offset 15, and occupies four columns.
  Loc loc = {.filename = "foo.carbon",
             .line = "  var x: i32 = \xe4\xb8\x96\xe7\x95\x8c;",
             .line_number = 3,
             .column_number = 16,
             .length = 6};
  EXPECT_THAT(RenderOne(capabilities, Level::Error, loc, "not found"),
              Eq("       ╭─┤ foo.carbon:3:16\n"
                 "->   3 │   var x: i32 = 世界;\n"
                 "       ·                ━━┳━\n"
                 "       ·                  ┗━━┫ error: not found\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, SpanPastEndOfLineIsClamped) {
  Loc loc = {.filename = "foo.carbon",
             .line = "var",
             .line_number = 1,
             .column_number = 40,
             .length = 10};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "at end"),
              Eq("       .-| foo.carbon:1:40\n"
                 "->   1 | var\n"
                 "       :    ^\n"
                 "       :    '--| error: at end\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, MergedShowsOneViewOfTheFile) {
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "too many arguments"),
      .labels = {
          MakeLabel(LabelCategory::Info, At(1, 1, 12), "declared here")}};
  // The note is above the error in the file, so it is drawn first, and the two
  // lines between them are elided.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "     1 | fn Run0() {}\n"
                 "       : ------.-----\n"
                 "       :       '--| declared here\n"
                 "       :\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: too many arguments\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, MergedShowsASingleSkippedLine) {
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(3, 1, 2), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 2), "here")}};
  // Only line 2 is between them, so it is shown rather than elided.
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("       .-| foo.carbon:3:1\n"
                                              "     1 | fn Run0() {}\n"
                                              "       : -.\n"
                                              "       :  '--| here\n"
                                              "     2 |\n"
                                              "->   3 | fn Main() {\n"
                                              "       : ^^\n"
                                              "       :  '--| error: bad\n"
                                              "       '----\n"
                                              ""));
}

TEST(RendererTest, NotesInAnotherFileGetTheirOwnLocationRow) {
  Loc other = {.filename = "bar.carbon",
               .line = "class C {",
               .line_number = 9,
               .column_number = 1,
               .length = 9};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, other, "declared here")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: bad\n"
                 "       :\n"
                 "       |-| bar.carbon:9:1\n"
                 "     9 | class C {\n"
                 "       : ----.----\n"
                 "       :     '--| declared here\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, APrimaryLabelTeesIntoItsRange) {
  // The label line leaves the middle of the underline through a junction, so
  // the underline and what it says are one line and the words point into the
  // range rather than off an end of it.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message =
          MakeMessage(At(4, 3, 7), "1 argument passed to function expecting 0"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7),
                           "1 argument passed here")}};
  EXPECT_THAT(
      Render(Plain(), diagnostic),
      Eq("       .-| foo.carbon:4:3\n"
         "->   4 |   Run0(1);\n"
         "       :   ^^^^^^^\n"
         "       :      '--| error: 1 argument passed to function expecting 0\n"
         "       :      '----| 1 argument passed here\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, TheShapeTheDesignDocDraws) {
  // The anatomy illustration in /toolchain/docs/diagnostics_rendering.md, so
  // that the picture there is output this produced rather than one drawn by
  // hand and left behind by the next change.
  static constexpr llvm::StringLiteral LibFile = "fn Run0() {}\n";
  Loc declaration = {.filename = "lib.carbon",
                     .line = "fn Run0() {}",
                     .file_text = LibFile,
                     .line_number = 1,
                     .column_number = 1,
                     .length = 12};
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(
          At(4, 3, 7), "1 argument passed to function expecting 0 arguments"),
      .labels = {MakeLabel(LabelCategory::Info, declaration,
                           "calling function declared here")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: 1 argument passed to function "
                 "expecting 0 arguments\n"
                 "       ┆\n"
                 "       ├─┤ lib.carbon:1:1\n"
                 "     1 │ fn Run0() {}\n"
                 "       · ──────┬─────\n"
                 "       ·       ╰──┤ calling function declared here\n"
                 "       ╰────\n"
                 ""));

  // The same diagnostic on a terminal with neither color nor Unicode, which is
  // the document's degradation example.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: 1 argument passed to function "
                 "expecting 0 arguments\n"
                 "       :\n"
                 "       |-| lib.carbon:1:1\n"
                 "     1 | fn Run0() {}\n"
                 "       : ------.-----\n"
                 "       :       '--| calling function declared here\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AFileOnlyLabelIsPlacedUnderItsLocationRow) {
  // A label whose location names only a file has its text placed under a
  // location row of its own, which an ordinary diagnostic doesn't need.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 8, 1), "cannot copy argument of type `A`"),
      .labels = {MakeLabel(LabelCategory::Info, FileOnly(),
                           "type `A` does not implement interface "
                           "`Core.Copy`")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       ╭─┤ foo.carbon:4:8\n"
         "->   4 │   Run0(1);\n"
         "       ·        ┳\n"
         "       ·        ┗━━┫ error: cannot copy argument of type `A`\n"
         "       ┆\n"
         "       ├─┤ foo.carbon\n"
         "       · ╰──┤ type `A` does not implement interface `Core.Copy`\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ARangeContainingTheMessageLocHostsIt) {
  // A range containing the message's location is drawn over the code that
  // location names, so it stands in for it and the location has no underline of
  // its own.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), "")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: bad call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ARangeElsewhereOnTheLineDoesNotHostTheMessage) {
  // Only a range containing the message's location takes its place. A range
  // elsewhere on the same line is not about the code the location names, so
  // the message keeps its own underline and text.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 8, 3), "")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^    ---\n"
                 "       :   '--| error: bad call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, RepeatedSpansAreOneUnderline) {
  // Two labels attached to the same range draw the same underline in the same
  // columns, so they are one underline rather than two identical rows. A
  // message that a context leads and the range attached to it reach this shape.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), ""),
                 MakeLabel(LabelCategory::Primary, At(4, 3, 7), "wrong")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: bad call\n"
                 "       :      '----| wrong\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ARangeCoveringOthersIsRepairedRatherThanStacked) {
  // A range attached over the whole expression, with ranges attached to the
  // operands inside it, is the shape overlap actually arrives in. Nothing is in
  // a position to reject it, so it is drawn as one row: the covering range is
  // the heavy underline and stays whole, and each operand shows as the tee
  // where its label line leaves that underline -- which ASCII keeps as the
  // corner inside the `^` run, since the `^` would hide where the label line
  // left.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), ""),
                 MakeLabel(LabelCategory::Info, At(4, 3, 4), "callee"),
                 MakeLabel(LabelCategory::Info, At(4, 8, 1), "argument")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^.^^.^\n"
                 "       :     |'----| error: bad call\n"
                 "       :     '-------| callee\n"
                 "       :        '------| argument\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ALabelOnASingleColumnRangeDropsStraightDown) {
  // With one column underlined there is no underline to turn out of, and
  // reaching in from the side would read as attaching to the column beside it.
  // The label line drops straight down the underline's own column, teeing into
  // it.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 8, 1), "bad index"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 4), "indexed here"),
                 MakeLabel(LabelCategory::Primary, At(4, 8, 1), "with this")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:8\n"
                 "->   4 |   Run0(1);\n"
                 "       :   --.- ^\n"
                 "       :     |  '--| error: bad index\n"
                 "       :     '-------| indexed here\n"
                 "       :        '------| with this\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, OverlappingSpansStillShareOneRow) {
  // Ranges covering some of the same columns are still one underline row. The
  // wider one here is the heavy underline, drawn last, so a range inside it
  // with no text of its own shows nothing.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), ""),
                 MakeLabel(LabelCategory::Info, At(4, 7, 3), "")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: bad call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ARangeInsideTheHeavyUnderlineShowsAsItsTee) {
  // The heavy underline is drawn last, so a narrower range inside it leaves it
  // whole, with no stretch of it light or in a ramp color, and the range shows
  // as the tee where its label line leaves the heavy underline. Over any other
  // underline, the narrower range is drawn on top.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 3, 4), "callee"),
                 MakeLabel(LabelCategory::Info, At(4, 8, 1), "argument")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━┯┳━┯━\n"
                 "       ·     │┗━━━━┫ error: bad call\n"
                 "       ·     ╰───────┤ callee\n"
                 "       ·        ╰──────┤ argument\n"
                 "       ╰────\n"
                 ""));
}

// The glyphs an underline row must never hold. A corner reads as an underline
// turning into the label line leaving it, a crossing as two underlines meeting,
// and an upward tee as an underline joining the source above it. None of those
// is ever what happened, whatever the ranges on the row are.
static constexpr std::array<llvm::StringLiteral, 6> ForbiddenInUnderlines = {
    "╭", "╮", "╯", "╰", "┼", "┴"};

// The ASCII stand-ins for the same shapes: `'` is a corner arriving from
// above and `+` a crossing. `.`, the corner leaving downward, is left out
// because it is also how a point with no room for anything else degrades.
static constexpr std::array<llvm::StringLiteral, 2> ForbiddenInAsciiUnderlines =
    {"'", "+"};

// Returns the underline row under the source, which is the row after it.
static auto UnderlineRow(llvm::StringRef rendered) -> std::string {
  llvm::SmallVector<llvm::StringRef> rows;
  rendered.split(rows, '\n');
  for (auto [index, row] : llvm::enumerate(rows)) {
    if (row.contains("abcdefghij") && index + 1 < rows.size()) {
      return rows[index + 1].str();
    }
  }
  return "";
}

TEST(RendererTest, UnderlinesNeverHoldACornerOrACrossing) {
  // Adjacency moves the ends of underlines and the columns label lines leave
  // them from, which is where a shape nothing drew could come from. This sweeps
  // what ranges on one row can be rather than naming the combinations one at a
  // time.
  static constexpr llvm::StringLiteral Line = "  abcdefghijklmnopqrstuvwxyz";
  auto at = [&](int column, int length) {
    return Loc{.filename = "f.carbon",
               .line = Line,
               .line_number = 1,
               .column_number = column,
               .length = length};
  };
  struct Placed {
    int start;
    int length;
    bool has_words;
  };
  auto render = [&](llvm::ArrayRef<Placed> ranges, Terminal::Charset charset,
                    LabelCategory category) {
    Terminal::Capabilities capabilities = {.charset = charset};
    llvm::SmallVector<Label> labels;
    for (const Placed& range : ranges) {
      labels.push_back(MakeLabel(category, at(range.start, range.length),
                                 range.has_words ? "words" : ""));
    }
    // The message points past every range, so each underline comes from a
    // label.
    Diagnostic diagnostic = {.level = Level::Error,
                             .message = MakeMessage(at(26, 1), "m"),
                             .labels = std::move(labels)};
    Renderer renderer(capabilities);
    llvm::SmallString<256> bytes;
    renderer.Render(bytes, diagnostic);
    return std::string(bytes);
  };
  auto check = [&](Terminal::Charset charset, llvm::StringRef what,
                   llvm::StringRef rendered) {
    std::string underlines = UnderlineRow(rendered);
    // An empty underline row means the source row went missing, which would let
    // every check below pass over nothing.
    EXPECT_THAT(underlines, Not(testing::IsEmpty())) << what << "\n"
                                                     << rendered;
    auto forbidden_glyphs =
        charset == Terminal::Charset::Utf8
            ? llvm::ArrayRef<llvm::StringLiteral>(ForbiddenInUnderlines)
            : llvm::ArrayRef<llvm::StringLiteral>(ForbiddenInAsciiUnderlines);
    for (llvm::StringLiteral forbidden : forbidden_glyphs) {
      EXPECT_THAT(underlines, Not(HasSubstr(forbidden))) << what << "\n"
                                                         << rendered;
    }
  };

  for (Terminal::Charset charset :
       {Terminal::Charset::Utf8, Terminal::Charset::Ascii}) {
    for (LabelCategory category :
         {LabelCategory::Info, LabelCategory::Primary}) {
      // Two and three ranges laid end to end and one column apart, in every
      // combination of lengths and of which carry words.
      for (int gap : {0, 1}) {
        for (int first : {1, 2, 3, 4}) {
          for (int second : {1, 2, 3, 4}) {
            for (int words = 0; words < 4; ++words) {
              check(charset, "two ranges",
                    render({{3, first, (words & 1) != 0},
                            {3 + first + gap, second, (words & 2) != 0}},
                           charset, category));
            }
          }
        }
        for (int first : {1, 2}) {
          for (int middle : {1, 2}) {
            for (int last : {1, 2}) {
              for (int words = 0; words < 8; ++words) {
                check(charset, "three ranges",
                      render({{3, first, (words & 1) != 0},
                              {3 + first + gap, middle, (words & 2) != 0},
                              {3 + first + middle + 2 * gap, last,
                               (words & 4) != 0}},
                             charset, category));
              }
            }
          }
        }
      }
      // Ranges covering each other, which layout repairs rather than refuses.
      for (int words = 0; words < 8; ++words) {
        check(charset, "a range covering others",
              render({{3, 6, (words & 1) != 0},
                      {4, 1, (words & 2) != 0},
                      {5, 1, (words & 4) != 0}},
                     charset, category));
        check(charset, "ranges covering one column",
              render({{3, 1, (words & 1) != 0},
                      {3, 1, (words & 2) != 0},
                      {4, 1, (words & 4) != 0}},
                     charset, category));
      }
    }
  }
}

TEST(RendererTest, AdjacentRangesAreDrawnApart) {
  // Ranges that touch have nothing between them to say where one ends and the
  // next begins, so the ends that meet stop at the centers of their cells and
  // leave a column of gap, half taken from each.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 7, 3), "with these")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━┳╸╶┬─\n"
                 "       ·     ┗━━━━━┫ error: bad call\n"
                 "       ·        ╰────┤ with these\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALabelLineWithNothingToTeeIntoTakesTheUnderlinesColumn) {
  // A single column with a range beside it has nowhere a label line could leave
  // the underline without the underline reaching the range next to it. The
  // label line takes the column instead of teeing into it, leaving from the
  // center of the cell: the column is underlined, and nothing is claimed of the
  // row above it.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 7, 1), "this one")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━┳╸╷\n"
                 "       ·     ┗━━━━┫ error: bad call\n"
                 "       ·       ╰────┤ this one\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALabelLineInPlaceOfAnUnderlineKeepsTheCaretUnderAscii) {
  // Where the message's underline is one column with an end pulled in to clear
  // the range beside it, there is no junction, so the label line is drawn down
  // that column in place of the underline. Its top cell is still part of the
  // heavy underline, so under ASCII it is `^`, not the `|` of a light label
  // line, and the rest of the label line is light like any label's.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 7, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 8, 2), "operand")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:7\n"
                 "->   4 |   Run0(1);\n"
                 "       :       ^-.\n"
                 "       :       '----| error: bad call\n"
                 "       :         '----| operand\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AdjacentLabelLinesAreTwoUnderlinesRatherThanOne) {
  // Two single columns side by side each have their label line in place of an
  // underline, and a label line leaving from the center of its cell is what
  // keeps the pair reading as two underlines: running them the height of their
  // cells would draw one unbroken line two columns wide.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 7, 1), "left one"),
                 MakeLabel(LabelCategory::Info, At(4, 8, 1), "right one")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━┳╸╷╷\n"
                 "       ·     ┗━━━━━┫ error: bad call\n"
                 "       ·       ╰─────┤ left one\n"
                 "       ·        ╰──────┤ right one\n"
                 "       ╰────\n"
                 ""));

  // ASCII draws no half cell, so a line stopping at one is `|` there like any
  // other: the pair reads as two underlines only where the character set can
  // draw the halves.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^||\n"
                 "       :     '-----| error: bad call\n"
                 "       :       '-----| left one\n"
                 "       :        '------| right one\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ALabelLineStepsAwayFromAnAdjacentRangeToTee) {
  // Two columns leave the label line one to tee into once the end beside the
  // next range is pulled in, so it steps inward rather than turning a corner
  // where the underline stops. The label follows it.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 5, 2), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 3, 2), "and this")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:5\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ┬╴╺┳\n"
                 "       ·   │  ┗━━┫ error: bad call\n"
                 "       ·   ╰───────┤ and this\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, AGapGivesWayToAnUnderlineThatCannotSpareIt) {
  // The gap is worth having but not worth paying for. Two columns with a range
  // on either side can give up one end but not both, since the label line needs
  // a column the underline still runs out through on both sides. The end beside
  // the range that begins after it keeps its column, and that range gives way
  // instead, so the boundary still shows a gap.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 10, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 3, 2), ""),
                 MakeLabel(LabelCategory::Info, At(4, 5, 2), "middle"),
                 MakeLabel(LabelCategory::Info, At(4, 7, 2), "")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:10\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ─╴╶┬╶─ ┳\n"
                 "       ·      │   ┗━━┫ error: bad call\n"
                 "       ·      ╰────────┤ middle\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ARangeWithNeighborsOnBothSidesAndNoLabelIsAPoint) {
  // A single column between two ranges has both its ends pulled in, and with no
  // words to hang there is no label line to draw down it either. The underline
  // is a point: nothing that fits one column shows a gap on each side of it.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 7, 1), ""),
                 MakeLabel(LabelCategory::Info, At(4, 8, 1), "this one")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━┳╸·╷\n"
                 "       ·     ┗━━━━━┫ error: bad call\n"
                 "       ·        ╰────┤ this one\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, AsciiDrawsAdjacentRangesWithoutTheGap) {
  // Half a cell is not something the ASCII stand-ins can draw, so the gap is
  // one of the distinctions that character set costs. The underlines still run
  // the width of their ranges.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 7, 3), "with these")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^-.-\n"
                 "       :     '-----| error: bad call\n"
                 "       :        '----| with these\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AnInformationalRangeDoesNotHostTheMessage) {
  // Only a range that is part of the problem stands in for the message's
  // location. A note covering the same code is a different underline, in the
  // note's own color, and the message's own location is still underlined.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 3, 7), "")}};
  // Rendered in color, since with none the two underlines are the same
  // characters and the whole distinction is which color each is drawn in.
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  std::string rendered = Render(capabilities, diagnostic);
  // The note's underline runs the width of its range in the note color, and
  // the message's own column is drawn over it in the level's.
  EXPECT_THAT(rendered, HasSubstr("\x1b[96m"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[91m"));
}

TEST(RendererTest, TwoSpansOnOneLineShareItsRow) {
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(1, 4, 4), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 2), "here")}};
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("       .-| foo.carbon:1:4\n"
                                              "->   1 | fn Run0() {}\n"
                                              "       : -. ^^^^\n"
                                              "       :  |   '--| error: bad\n"
                                              "       :  '--------| here\n"
                                              "       '----\n"
                                              ""));
}

TEST(RendererTest, LabelsOnOneLineReadInSourceOrder) {
  // The rows read top to bottom in the order the code reads left to right:
  // the message first, then each label in the order of its range. Each
  // label's text starts two columns right of the text above it and right of
  // every label line's column, so a label line descending to a later label runs
  // beside the text above it and breaks only for single cells of the runs it
  // crosses.
  static constexpr llvm::StringLiteral Line = "  return count * flag;";
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = Line,
               .line_number = 3,
               .column_number = column,
               .length = length};
  };
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(16, 1), "`i32` does not implement `MulWith`"),
      .labels = {MakeLabel(LabelCategory::Primary, at(10, 5),
                           "left operand has type `i32`"),
                 MakeLabel(LabelCategory::Primary, at(18, 4),
                           "right operand has type `bool`")}};
  EXPECT_THAT(
      Render(Plain(), diagnostic),
      Eq("       .-| foo.carbon:3:16\n"
         "->   3 |   return count * flag;\n"
         "       :          --.-- ^ --.-\n"
         "       :            |   '------| error: `i32` does not implement "
         "`MulWith`\n"
         "       :            '------------| left operand has type `i32`\n"
         "       :                    '------| right operand has type `bool`\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, SpansOnOneLineShareOneWindow) {
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  // Both spans are drawn against the one row that shows the line, so both have
  // to be measured in the window that row was cut to.
  std::string line(200, 'x');
  line.replace(20, 5, "FIRST");
  line.replace(40, 6, "SECOND");
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = line,
               .line_number = 1,
               .column_number = column,
               .length = length};
  };
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(21, 5), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, at(41, 6), "and here")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       .-| foo.carbon:1:21\n"
         "->   1 | "
         "...xxxxxxxxxxxxxxxxFIRSTxxxxxxxxxxxxxxxSECONDxxxxxxxxxxxxxxxxxxxxxxx."
         "..\n"
         "       :                    ^^^^^               ---.--\n"
         "       :                      '-----------------------| error: bad\n"
         "       :                                           '----| and here\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, MultipleFilesEachGetALocationRow) {
  Loc first = {.filename = "b.carbon",
               .line = "fn Shape() {}",
               .line_number = 3,
               .column_number = 1,
               .length = 13};
  Loc second = {.filename = "a.carbon",
                .line = "class Shape {}",
                .line_number = 3,
                .column_number = 1,
                .length = 14};
  Diagnostic diagnostic = {.level = Level::Error,
                           .message = MakeMessage(first, "duplicate name"),
                           .labels = {MakeLabel(LabelCategory::Info, second,
                                                "previously declared")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| b.carbon:3:1\n"
                 "->   3 | fn Shape() {}\n"
                 "       : ^^^^^^^^^^^^^\n"
                 "       :       '--| error: duplicate name\n"
                 "       :\n"
                 "       |-| a.carbon:3:1\n"
                 "     3 | class Shape {}\n"
                 "       : -------.------\n"
                 "       :        '--| previously declared\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ALabelsPathIsNotDrawn) {
  Loc import_loc = {.filename = "c.carbon",
                    .line = "import library \"a\";",
                    .line_number = 1,
                    .column_number = 1,
                    .length = 6};
  Loc other = {.filename = "a.carbon",
               .line = "class Shape {}",
               .line_number = 3,
               .column_number = 1,
               .length = 14};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "duplicate name"),
      .labels = {MakeLabel(LabelCategory::Info, other, "previously declared",
                           {MakeOrigin(import_loc, "in import")})}};
  // A label's path is not drawn, so its location row has no path rows above it.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: duplicate name\n"
                 "       :\n"
                 "       |-| a.carbon:3:1\n"
                 "     3 | class Shape {}\n"
                 "       : -------.------\n"
                 "       :        '--| previously declared\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, APathLeadsIntoTheLocationRowItReaches) {
  Loc import_loc = {.filename = "c.carbon",
                    .line = "import library \"a\";",
                    .line_number = 1,
                    .column_number = 1,
                    .length = 6};
  Loc reported = {.filename = "a.carbon",
                  .line = "class Shape {}",
                  .line_number = 3,
                  .column_number = 1,
                  .length = 14};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(reported, "duplicate name",
                             {MakeOrigin(import_loc, "in import")}),
      .contexts = {MakeContext(At(4, 3, 7), "in call")}};
  // The message's own path is drawn, above the location row of the file it
  // leads to -- here the second location row, since the context's snippet comes
  // first.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: in call\n"
                 "       :\n"
                 "       | .-- in import: c.carbon:1:1\n"
                 "       |-| a.carbon:3:1\n"
                 "     3 | class Shape {}\n"
                 "       : -------.------\n"
                 "       :        '--| duplicate name\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ContextStepsStackInOrder) {
  auto step = [](llvm::StringRef file, int line) {
    return Loc{.filename = file, .line_number = line, .column_number = 1};
  };
  Loc target = {.filename = "inner.h",
                .line = "void Use(int, int);",
                .line_number = 7,
                .column_number = 6,
                .length = 3};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message =
          MakeMessage(target, "no matching function",
                      {MakeOrigin(step("main.carbon", 4), "included from"),
                       MakeOrigin(step("outer.h", 3), "included from")})};
  // Each step leads to the next, and the last leads into the location row, so
  // the path reads in the order it was walked.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       | .-- included from: main.carbon:4:1\n"
                 "       | |-- included from: outer.h:3:1\n"
                 "       |-| inner.h:7:6\n"
                 "->   7 | void Use(int, int);\n"
                 "       :      ^^^\n"
                 "       :       '--| error: no matching function\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, LocationInfoHasNoSnippet) {
  Loc import_loc = {.filename = "a.carbon",
                    .line = "import b;",
                    .line_number = 2,
                    .column_number = 1,
                    .length = 6};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "duplicate name",
                             {MakeOrigin(import_loc, "in import")})};
  // The location information sits above the location row of the location it
  // describes, and its own location follows its words.
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       | .-- in import: a.carbon:2:1\n"
                 "       |-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :      '--| error: duplicate name\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AFileOnlyLeadingMessageGivesItsFileOneLocationRow) {
  // A message located at a file and nothing more is placed under that file's
  // location row, and the file's spans follow under the same location row below
  // an elision, so the file has one location row.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(FileOnly(), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 2), "declared here"),
                 MakeLabel(LabelCategory::Info, At(4, 3, 7), "used here")}};
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("->     .-| foo.carbon\n"
                                              "       : '--| error: bad\n"
                                              "       :\n"
                                              "     1 | fn Run0() {}\n"
                                              "       : -.\n"
                                              "       :  '--| declared here\n"
                                              "       :\n"
                                              "     4 |   Run0(1);\n"
                                              "       :   ---.---\n"
                                              "       :      '--| used here\n"
                                              "       '----\n"
                                              ""));
}

TEST(RendererTest, FirstContextMessageTakesTheDiagnosticLevel) {
  Diagnostic diagnostic = {.level = Level::Error,
                           .message = MakeMessage({}, "the problem"),
                           .contexts = {MakeContext({}, "in call"),
                                        MakeContext({}, "in other call")}};
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("error: in call\n"
                                              "note: in other call\n"
                                              "note: the problem\n"));
}

TEST(RendererTest, KindIsIncludedWhenAsked) {
  Renderer renderer(Plain());
  renderer.set_include_kind(true);
  llvm::SmallString<256> bytes;
  renderer.Render(bytes,
                  {.level = Level::Error, .message = MakeMessage({}, "oops")});
  EXPECT_THAT(std::string(bytes), Eq("error: oops [TestDiagnostic]\n"));
}

TEST(RendererTest, Utf8DrawsTheFrameWithBoxDrawing) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 12), "here")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "     1 │ fn Run0() {}\n"
                 "       · ──────┬─────\n"
                 "       ·       ╰──┤ here\n"
                 "       ┆\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: bad\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, Utf8ElisionContinuesTheFrameToTheNextLocationRow) {
  // The step from one snippet to the next file's location row is an elision, as
  // skipped lines inside a snippet are, so the frame continues unbroken down
  // to the next location row's `├`.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  static constexpr llvm::StringLiteral LibFile = "fn Run0() {}\n";
  Loc declaration = {.filename = "lib.carbon",
                     .line = "fn Run0() {}",
                     .file_text = LibFile,
                     .line_number = 1,
                     .column_number = 1,
                     .length = 12};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "too many arguments"),
      .labels = {MakeLabel(LabelCategory::Info, declaration, "declared here")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: too many arguments\n"
                 "       ┆\n"
                 "       ├─┤ lib.carbon:1:1\n"
                 "     1 │ fn Run0() {}\n"
                 "       · ──────┬─────\n"
                 "       ·       ╰──┤ declared here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, Utf8ContextStepsStackIntoOneLine) {
  auto step = [](llvm::StringRef file, int line) {
    return Loc{.filename = file, .line_number = line, .column_number = 1};
  };
  Loc target = {.filename = "inner.h",
                .line = "void Use(int, int);",
                .line_number = 7,
                .column_number = 6,
                .length = 3};
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message =
          MakeMessage(target, "no matching function",
                      {MakeOrigin(step("main.carbon", 4), "included from"),
                       MakeOrigin(step("outer.h", 3), "included from")})};
  // Each step's line descends into the next, so the first opens the line and
  // the rest are tees on it, and the last descends into the bracket that opens
  // the location row. All of that is junctions the buffer forms where the
  // segments meet; none of these glyphs is named by the renderer.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╷ ╭── included from: main.carbon:4:1\n"
                 "       │ ├── included from: outer.h:3:1\n"
                 "       ├─┤ inner.h:7:6\n"
                 "->   7 │ void Use(int, int);\n"
                 "       ·      ━┳━\n"
                 "       ·       ┗━━┫ error: no matching function\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, Utf8LocationLabelIsPlacedUnderItsLocationRow) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, FileOnly(), "declared here")}};
  // A label with no source has its text placed under a location row naming what
  // location it has, and an elision separates that location row from the
  // snippet above, as for any other location row.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: bad\n"
                 "       ┆\n"
                 "       ├─┤ foo.carbon\n"
                 "       · ╰──┤ declared here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALineWithoutAColumnIsPlacedUnderItsLocationRow) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, LineOnly(1), "declared here")}};
  // A location with a line and no column has no source, so the label's text
  // is placed under a location row naming the line, as for one naming only a
  // file.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: bad\n"
                 "       ┆\n"
                 "       ├─┤ foo.carbon:1\n"
                 "       · ╰──┤ declared here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, APartWithNoLocationGetsANoLocationRow) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, Loc(), "declared here")}};
  // A part with no location at all still gets a location row for its text,
  // which names `<no location>`.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: bad\n"
                 "       ┆\n"
                 "       ├─┤ <no location>\n"
                 "       · ╰──┤ declared here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALeadingMessageWithNoLocationOpensTheFrame) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(Loc(), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 2), "declared here")}};
  // A message with no location still leads: its text is placed under a
  // `<no location>` location row with a heavy label line and bar, the location
  // row takes the `->`, and the label's file gets a location row after it.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("->     ╭─┤ <no location>\n"
                 "       · ┗━━┫ error: bad\n"
                 "       ┆\n"
                 "       ├─┤ foo.carbon:1:1\n"
                 "     1 │ fn Run0() {}\n"
                 "       · ─┬\n"
                 "       ·  ╰──┤ declared here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, Utf8WrappedLocationLabelKeepsItsBar) {
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 76};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad"),
      .labels = {MakeLabel(
          LabelCategory::Info, FileOnly(),
          "this one has rather a lot to say about the declaration it points "
          "at, more than fits a row")}};
  // A location label wraps as a label does, with the bar down every row it
  // takes.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: bad\n"
                 "       ┆\n"
                 "       ├─┤ foo.carbon\n"
                 "       · ╰──┤ this one has rather a lot to say about the "
                 "declaration it\n"
                 "       ·    │ points at, more than fits a row\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ColorStylesEachElement) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  EXPECT_THAT(
      RenderOne(capabilities, Level::Error, At(4, 3, 7), "bad `x` here"),
      // Bold and the color arrive as separate sequences, and blanks between
      // styled runs carry no style of their own. The message is one style
      // throughout, backticks included.
      Eq("       \x1b[1m\x1b[94m.-|\x1b[0m foo.carbon:4:3\n"
         "\x1b[1m\x1b[91m->\x1b[0m   \x1b[1m\x1b[91m4\x1b[0m "
         "\x1b[1m\x1b[94m|\x1b[0m   Run0(1);\n"
         "       \x1b[91m:\x1b[0m   \x1b[1m\x1b[91m^^^^^^^\n"
         "\x1b[0m       \x1b[91m:\x1b[0m      \x1b[1m\x1b[91m'--|\x1b[0m "
         "\x1b[1m\x1b[91merror: \x1b[0m\x1b[1mbad `x` here\n"
         "\x1b[0m       \x1b[1m\x1b[94m'----\x1b[0m\n"
         ""));
}

TEST(RendererTest, UnderlinesTakeTheColorOfTheirMessage) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Warning,
      .message = MakeMessage(At(4, 3, 7), "careful", {}, Level::Warning),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 12), "aside")}};
  std::string rendered = Render(capabilities, diagnostic);
  // The message's underline is the warning color and the note's is the note
  // color, so each underline matches the text that explains it. The message's
  // underline is heavy and bold, matching its bold text; the note's is neither,
  // since terminals render bold on line art unreliably.
  EXPECT_THAT(rendered, HasSubstr("\x1b[1m\x1b[93m^^^^^^^\n"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[96m------.-----\n"));
  // The line the problem is on takes the level's color and its weight. Every
  // other line number is as light as the frame it belongs to, so the one being
  // reported is the one that stands out.
  EXPECT_THAT(rendered, HasSubstr("\x1b[1m\x1b[93m4\x1b[0m"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[94m1\x1b[0m"));
}

TEST(RendererTest, TheFrameSeparatorIsFrame) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Warning,
      .message = MakeMessage(At(4, 3, 7), "careful", {}, Level::Warning),
      .labels = {MakeLabel(LabelCategory::Info, At(1, 1, 12), "aside")}};
  std::string rendered = Render(capabilities, diagnostic);
  // The frame's glyph on underline and label rows is in the frame's column, so
  // it is the leading part's color on every row, including the note's, whose
  // underline and label are the note color beside it.
  EXPECT_THAT(rendered, HasSubstr("\x1b[93m:\x1b[0m \x1b[96m------.-----"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[93m:\x1b[0m       \x1b[96m'--|"));
}

TEST(RendererTest, RotateColorsEachRangeOnALine) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 4), ""),
                 MakeLabel(LabelCategory::Primary, At(4, 8, 1), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  // The range hosting the message keeps the theme's center color, heavy and
  // bold. The other problem range takes the ramp with the center left out, so
  // it gets the dark end. Both are reds, so the color shows the theme first.
  EXPECT_THAT(rendered, HasSubstr("\x1b[1m\x1b[91m^^^^"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[31m-"));
}

TEST(RendererTest, RotateStaysInsideAThemeWithOnlyNamedColors) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 8, 1), ""),
                 MakeLabel(LabelCategory::Info, At(4, 9, 1), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  // The ranges explaining the problem walk their own ramp, so a second note
  // range is another cyan rather than the level's second color.
  EXPECT_THAT(rendered, HasSubstr("\x1b[36m-"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[96m-"));
}

TEST(RendererTest, TruecolorWarningMatchesTheErrorLuminance) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Truecolor};
  // Bright yellow is four times the luminance of bright red, which puts a
  // warning above an error on the one axis a reader takes as importance. With
  // 24-bit color the colors are ours to choose, so the three centers sit at one
  // lightness and the level is said by hue rather than by loudness.
  EXPECT_THAT(RenderOne(capabilities, Level::Warning, At(4, 3, 7), "careful"),
              HasSubstr("\x1b[38;2;172;172;0m"));
  EXPECT_THAT(RenderOne(capabilities, Level::Error, At(4, 3, 7), "bad"),
              HasSubstr("\x1b[38;2;255;114;97m"));
}

TEST(RendererTest, LightBackgroundGetsItsOwnColors) {
  Terminal::Capabilities dark = {.color_mode = Terminal::ColorMode::Truecolor};
  Terminal::Capabilities light = {.color_mode = Terminal::ColorMode::Truecolor,
                                  .background = Terminal::Background::Light};
  // A color picked to read against black is hard to read against white, so the
  // two get separate palettes rather than one compromise.
  EXPECT_THAT(RenderOne(light, Level::Error, At(4, 3, 7), "bad"),
              HasSubstr("\x1b[38;2;208;0;0m"));
  EXPECT_THAT(RenderOne(dark, Level::Error, At(4, 3, 7), "bad"),
              HasSubstr("\x1b[38;2;255;114;97m"));
}

TEST(RendererTest, LightBackgroundDoesNotMoveNamedColors) {
  Terminal::Capabilities light = {.color_mode = Terminal::ColorMode::Ansi16,
                                  .background = Terminal::Background::Light};
  // A named color is drawn from a palette the user chose to go with their
  // background, so there is nothing here to adapt.
  EXPECT_THAT(RenderOne(light, Level::Error, At(4, 3, 7), "bad"),
              HasSubstr("\x1b[91m"));
}

TEST(RendererTest, TruecolorRotatesInsideATheme) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Truecolor};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 4), ""),
                 MakeLabel(LabelCategory::Primary, At(4, 8, 1), ""),
                 MakeLabel(LabelCategory::Info, At(1, 1, 12), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  // Three reds for the problem and a cyan for what explains it: every color
  // says which theme it belongs to before it says which range it is.
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;255;114;97m"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;181;86;0m"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;0;188;188m"));
}

TEST(RendererTest, RotateLeavesOneRangeOfAKindAlone) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad call"),
      .labels = {
          MakeLabel(LabelCategory::Info, At(1, 1, 12), "declared here")}};
  // A range alone on its line takes its theme's own color; the ramp's other
  // entries only appear once several ranges share a line.
  std::string rendered = Render(capabilities, diagnostic);
  EXPECT_THAT(rendered, HasSubstr("\x1b[96m"));
  EXPECT_THAT(rendered, Not(HasSubstr("\x1b[36m")));
}

TEST(RendererTest, ContextIsDim) {
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Loc import_loc = {
      .filename = "c.carbon", .line_number = 1, .column_number = 1};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad",
                             {MakeOrigin(import_loc, "imported from")})};
  // The words and the location are one dim run, so that the whole row recedes
  // behind the code the diagnostic is really about.
  EXPECT_THAT(Render(capabilities, diagnostic),
              HasSubstr("\x1b[2mimported from: c.carbon:1:1\n"));
}

TEST(RendererTest, LongSourceLineIsWindowedOnATerminal) {
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  std::string line(200, 'x');
  line.replace(100, 4, "HERE");
  Loc loc = {.filename = "foo.carbon",
             .line = line,
             .line_number = 1,
             .column_number = 101,
             .length = 4};
  EXPECT_THAT(
      RenderOne(capabilities, Level::Error, loc, "found it"),
      Eq("       .-| foo.carbon:1:101\n"
         "->   1 | "
         "...xxxxxxxxxxxxxxxxHERExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx."
         "..\n"
         "       :                    ^^^^\n"
         "       :                      '--| error: found it\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, AFormattedSourceLineIsShownWhole) {
  // Nothing states a width here, so the target is the width code is formatted
  // to. A line inside it is shown whole: the point of deriving the target from
  // the source width rather than the terminal's is that formatted code is never
  // cut for want of a terminal to measure.
  std::string line(TargetSourceColumns, 'x');
  line.replace(40, 4, "HERE");
  Loc loc = {.filename = "foo.carbon",
             .line = line,
             .line_number = 1,
             .column_number = 41,
             .length = 4};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "found it"),
              HasSubstr("1 | " + line + "\n"));
}

TEST(RendererTest, ASourceLinePastTheTargetIsWindowed) {
  // Past the target it is windowed, the same as a line past a terminal's width:
  // the target is a width to fit, and this is what fitting one means.
  std::string line(size_t{4} * TargetSourceColumns, 'x');
  line.replace(200, 4, "HERE");
  Loc loc = {.filename = "foo.carbon",
             .line = line,
             .line_number = 1,
             .column_number = 201,
             .length = 4};
  std::string rendered = RenderOne(Plain(), Level::Error, loc, "found it");
  EXPECT_THAT(rendered, HasSubstr("HERE"));
  EXPECT_THAT(rendered, HasSubstr("..."));
  EXPECT_THAT(rendered, Not(HasSubstr(line)));
}

TEST(RendererTest, LineShownBetweenTwoSpansIsWindowedToo) {
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  // A line between two spans is shown rather than elided, so it has to be cut
  // to the terminal the same way the spans' own lines are. Left whole, it wraps
  // and every row below it lands a line lower than the one it annotates.
  std::string file = "short\n" + std::string(200, 'x') + "\nalso short\n";
  auto at = [&](int line_number) {
    llvm::StringRef rest = file;
    for (auto _ : llvm::seq(1, line_number)) {
      rest = rest.split('\n').second;
    }
    return Loc{.filename = "foo.carbon",
               .line = rest.split('\n').first,
               .file_text = file,
               .line_number = line_number,
               .column_number = 1,
               .length = 4};
  };
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(3), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, at(1), "and")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       .-| foo.carbon:3:1\n"
                 "     1 | short\n"
                 "       : --.-\n"
                 "       :   '--| and\n"
                 "     2 | "
                 "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                 "xxxx...\n"
                 "->   3 | also short\n"
                 "       : ^^^^\n"
                 "       :   '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, WindowingKeepsWideCharactersWhole) {
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 80};
  // Each `界` is two columns, so a window that cut one in half would put every
  // column after it in the wrong place.
  std::string line;
  for (auto _ : llvm::seq(40)) {
    line += "界";
  }
  line.replace(60, 3, "x");
  Loc loc = {.filename = "foo.carbon",
             .line = line,
             .line_number = 1,
             .column_number = 61,
             .length = 1};
  std::string rendered = RenderOne(capabilities, Level::Error, loc, "found it");
  Terminal::Metrics metrics(Terminal::Charset::Utf8);
  // The underline lands under the `x`, however the window fell. Both columns
  // are measured rather than compared as byte offsets, since every `界` before
  // them is three bytes and two columns.
  auto column_of = [&](llvm::StringRef prefix, llvm::StringRef glyph) {
    llvm::StringRef row = llvm::StringRef(rendered).split(prefix).second;
    row = row.take_until([](char c) { return c == '\n'; });
    size_t at = row.find(glyph);
    EXPECT_NE(at, llvm::StringRef::npos) << row.str();
    return metrics.Width(row.substr(0, at));
  };
  EXPECT_THAT(column_of("->   1 │", "x"), Eq(column_of("       ·", "┳")));
  // Every row fits the terminal, which needs each `界` kept whole.
  llvm::StringRef rest = rendered;
  while (!rest.empty()) {
    auto [row, tail] = rest.split('\n');
    EXPECT_LE(metrics.Width(row), 80) << row.str();
    rest = tail;
  }
}

TEST(RendererTest, AnUnstatedWidthIsStillATargetToWrapTo) {
  std::string text =
      "a message with rather a lot to say, more than would fit a terminal of "
      "the usual width, and so more than enough to wrap";

  // There is no unbounded width: with nothing stated the target is the source
  // width plus the gutter, and the message wraps to it like any other. What a
  // stated width changes is the number, not whether there is one.
  // Wrapped under itself, past the level word, so a continuation cannot read as
  // the start of another diagnostic.
  EXPECT_THAT(RenderOne(Plain(), Level::Error, At(4, 3, 7), text),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^^^^\n"
                 "       :   '--| error: a message with rather a lot to say, "
                 "more than would fit a terminal\n"
                 "       :      |        of the usual width, and so more than "
                 "enough to wrap\n"
                 "       '----\n"
                 ""));

  // A width that is stated is fitted, which is the whole of the difference
  // between knowing one and not.
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  std::string wrapped =
      RenderOne(capabilities, Level::Error, At(4, 3, 7), text);
  // The message wraps to the column after its level word, inside its bar.
  EXPECT_THAT(wrapped, HasSubstr("\n       "));
  llvm::StringRef rest = wrapped;
  while (!rest.empty()) {
    auto [row, tail] = rest.split('\n');
    EXPECT_LE(row.size(), 80U) << row.str();
    rest = tail;
  }
}

// A file whose one interesting line leaves a label nowhere to hang at a width
// that still holds a frame.
static constexpr llvm::StringLiteral RoutedFile =
    "fn Main() {\n"
    "  NeedsIt(\"definitely not the right type\");\n"
    "}\n";

// Returns the span `NeedsIt`'s argument occupies in `RoutedFile`.
static auto RoutedLoc() -> Loc {
  llvm::StringRef line = llvm::StringRef(RoutedFile).split('\n').second;
  return {.filename = "foo.carbon",
          .line = line.split('\n').first,
          .file_text = RoutedFile,
          .line_number = 2,
          .column_number = 11,
          .length = 30};
}

// Renders a diagnostic whose only message carries `label`, at `columns`.
static auto RenderLabel(int columns, llvm::StringRef label,
                        Terminal::Charset charset = Terminal::Charset::Utf8)
    -> std::string {
  Terminal::Capabilities capabilities = {
      .charset = charset, .is_terminal = true, .columns = columns};
  return Render(
      capabilities,
      {.level = Level::Error,
       .message = MakeMessage(RoutedLoc(), "bad"),
       .labels = {MakeLabel(LabelCategory::Primary, RoutedLoc(), label)}});
}

TEST(RendererTest, AKindTagTakesARowOfItsOwnWhenItCannotShareOne) {
  // The tag is wrapped as part of the text it names rather than drawn after it,
  // so it follows the words where they share a row and takes a row of its own
  // where they do not. Drawing it after the text would have to start where the
  // text ended, which is outside the block whenever a word overhangs one.
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 76};
  auto render = [&](llvm::StringRef label) {
    Renderer renderer(capabilities);
    renderer.set_include_kind(true);
    llvm::SmallString<512> bytes;
    renderer.Render(
        bytes,
        {.level = Level::Error,
         .message = MakeMessage(RoutedLoc(), "bad"),
         .labels = {MakeLabel(LabelCategory::Primary, RoutedLoc(), label)}});
    return std::string(bytes);
  };

  EXPECT_THAT(
      render("short label"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━\n"
         "       ·                          ┗━━┫ error: bad [TestDiagnostic]\n"
         "       ·                          ╰────┤ short label [TestLabel]\n"
         "       ╰────\n"
         ""));

  // The tag is drawn on its own rather than joined to the text, so it keeps its
  // own style in either placement, which the colorless goldens cannot show.
  Terminal::Capabilities colored = capabilities;
  colored.color_mode = Terminal::ColorMode::Ansi16;
  auto tag_is_dim = [&](llvm::StringRef label) {
    Renderer renderer(colored);
    renderer.set_include_kind(true);
    llvm::SmallString<512> bytes;
    renderer.Render(
        bytes,
        {.level = Level::Error,
         .message = MakeMessage(RoutedLoc(), "bad"),
         .labels = {MakeLabel(LabelCategory::Primary, RoutedLoc(), label)}});
    return llvm::StringRef(bytes).contains("\x1b[2m[TestLabel]") ||
           llvm::StringRef(bytes).contains("\x1b[2m [TestLabel]");
  };
  EXPECT_TRUE(tag_is_dim("short label"));
  EXPECT_TRUE(
      tag_is_dim("type `str` does not implement interface "
                 "`Core.ImplicitAs(i32)` at all, and it really will "
                 "not be doing so any time soon either, sorry"));

  // The space separating the tag from the text is dropped once the tag starts a
  // row, so it lines up with the block it names.
  EXPECT_THAT(
      render("type `str` does not implement interface "
             "`Core.ImplicitAs(i32)` at all, and it really will not "
             "be doing so any time soon either, sorry"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         "       ·           ┗━━┫ error: bad [TestDiagnostic]\n"
         "       ·           ╰────┤ type `str` does not implement interface\n"
         "       ·                │ `Core.ImplicitAs(i32)` at all, and it "
         "really will\n"
         "       ·                │ not be doing so any time soon either, "
         "sorry\n"
         "       ·                │ [TestLabel]\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, AQuotedRunWrapsWhole) {
  // Code a message quotes is one word to wrapping, so a type written with
  // spaces moves to the next row whole rather than splitting at a space
  // inside the backticks. Every wrapped block follows the rule: the message's,
  // a label's, and the routed decision that asks how wide the widest word is.
  EXPECT_THAT(RenderLabel(62, "cannot convert type `{.a: i32, .b: i32}` here"),
              Eq("       ╭─┤ foo.carbon:2:11\n"
                 "->   2 │   NeedsIt(\"definitely not the right type\");\n"
                 "       ·           ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                 "       ·           ┗━━┫ error: bad\n"
                 "       ·           ╰────┤ cannot convert type\n"
                 "       ·                │ `{.a: i32, .b: i32}` here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALabelIsFramedWhenItWraps) {
  // The bar frames every row the label takes, so what belongs to it is bounded
  // rather than left to be read off the indentation, and the label line meets
  // that bar on the label's first row, where its text begins.
  EXPECT_THAT(
      RenderLabel(76,
                  "type `str` does not implement interface "
                  "`Core.ImplicitAs(i32)` at all, and it really will not "
                  "be doing so any time soon either, sorry"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         "       ·           ┗━━┫ error: bad\n"
         "       ·           ╰────┤ type `str` does not implement interface\n"
         "       ·                │ `Core.ImplicitAs(i32)` at all, and it "
         "really will\n"
         "       ·                │ not be doing so any time soon either, "
         "sorry\n"
         "       ╰────\n"
         ""));

  // A label of one row is framed the same as a label of ten: the bar is a
  // separate line on each row rather than one line through them, so the cell
  // the label line arrives at is a tee and not the corner an end would be.
  EXPECT_THAT(RenderLabel(76, "type `str` does not implement it"),
              Eq("       ╭─┤ foo.carbon:2:11\n"
                 "->   2 │   NeedsIt(\"definitely not the right type\");\n"
                 "       ·           ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━\n"
                 "       ·                          ┗━━┫ error: bad\n"
                 "       ·                          ╰────┤ type `str` does not "
                 "implement it\n"
                 "       ╰────\n"
                 ""));

  // ASCII keeps each corner's own glyph, and the bar below the label line is
  // what says the second row belongs to the label above it.
  EXPECT_THAT(
      RenderLabel(76,
                  "type `str` does not implement interface "
                  "`Core.ImplicitAs(i32)` at all",
                  Terminal::Charset::Ascii),
      Eq("       .-| foo.carbon:2:11\n"
         "->   2 |   NeedsIt(\"definitely not the right type\");\n"
         "       :           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
         "       :           '--| error: bad\n"
         "       :           '----| type `str` does not implement interface\n"
         "       :                | `Core.ImplicitAs(i32)` at all\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, ALabelSlidesLeftBeforeItWraps) {
  // Label lines slide left inside their ranges before any text wraps. Here the
  // label and the message share the range, so both label lines slide to its
  // start. Only then does text wrap, and a label is out-dented only where its
  // widest word does not fit right of its label line.
  EXPECT_THAT(
      RenderLabel(76,
                  "type `str` does not implement interface "
                  "`Core.ImplicitAs(i32)` at all"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         "       ·           ┗━━┫ error: bad\n"
         "       ·           ╰────┤ type `str` does not implement interface\n"
         "       ·                │ `Core.ImplicitAs(i32)` at all\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ALabelWithNowhereToHangIsRoutedBack) {
  // The label's widest word doesn't fit beside either end of the range, so it
  // is out-dented past the label line and its own frame is what reaches back:
  // the top of the bar turns right along the row under the underline and runs
  // to the column the label line came down.
  EXPECT_THAT(
      RenderLabel(76,
                  "`Core.ImplicitAs("
                  "ExtremelyLongNamedGenericContainerTypeForTesting(i32))`"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━\n"
         "       ·                          ┗━━┫ error: bad\n"
         "       · ╭────────────────────────╯\n"
         "       · │ "
         "`Core.ImplicitAs(ExtremelyLongNamedGenericContainerTypeForTesting("
         "i32))`\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ARoutedLabelIsFramedDownEveryRow) {
  // Meeting the label at its top rather than its middle is what lets the bar
  // sit directly against the text, which is where the width this form exists to
  // find comes from.
  EXPECT_THAT(
      RenderLabel(
          76,
          "type `str` does not implement interface "
          "`Core.ImplicitAs(ExtremelyLongNamedGenericContainerTypeForTesting("
          "i32, ExtremelyLongNamedGenericContainerTypeForTesting(i32, i32)))` "
          "and it "
          "will not be doing so any time soon"),
      Eq("       ╭─┤ foo.carbon:2:11\n"
         "->   2 │   NeedsIt(\"definitely not the right type\");\n"
         "       ·           ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━\n"
         "       ·                          ┗━━┫ error: bad\n"
         "       · ╭────────────────────────╯\n"
         "       · │ type `str` does not implement interface\n"
         "       · │ "
         "`Core.ImplicitAs(ExtremelyLongNamedGenericContainerTypeForTesting("
         "i32, ExtremelyLongNamedGenericContainerTypeForTesting(i32, i32)))`\n"
         "       · │ and it will not be doing so any time soon\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, LabelLinesSlideForAnotherLabelsWords) {
  // Every label's text starts right of every label line's column, so the
  // message's text fits on one row here only because the label line of the
  // label to its right slides left within that label's range, and only as
  // far as the message's text needs, short of where the range starts.
  static constexpr llvm::StringLiteral Line = "  Run0(aVeryLongArgument);";
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = Line,
               .line_number = 4,
               .column_number = column,
               .length = length};
  };
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 60};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(3, 4), "call would overflow the result"),
      .labels = {MakeLabel(LabelCategory::Info, at(8, 17), "short")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       ╭─┤ foo.carbon:4:3\n"
         "->   4 │   Run0(aVeryLongArgument);\n"
         "       ·   ━━┳━ ──┬──────────────\n"
         "       ·     ┗━━━━━━━┫ error: call would overflow the result\n"
         "       ·          ╰────┤ short\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, RoutedLabelsStackInReadingOrder) {
  // Two labels whose widest words do not fit even with every label line slid
  // are each out-dented to the source's column, in reading order, while the
  // message stays under its underline. Each block's bar reaches back to its
  // label line, and the label line descending to the later block passes the
  // earlier block's reach-back row and skips its text rows.
  static constexpr llvm::StringLiteral Line = "  Run0(first, second);";
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = Line,
               .line_number = 4,
               .column_number = column,
               .length = length};
  };
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 60};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(3, 4), "bad call"),
      .labels = {
          MakeLabel(LabelCategory::Info, at(8, 5),
                    "`ExtremelyLongUnbreakableTypeNameForTestingNumberOne`"),
          MakeLabel(LabelCategory::Info, at(15, 6),
                    "`ExtremelyLongUnbreakableTypeNameForTestingNumberTwo`")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       ╭─┤ foo.carbon:4:3\n"
         "->   4 │   Run0(first, second);\n"
         "       ·   ━━┳━ ──┬──  ───┬──\n"
         "       ·     ┗━━━━━━━━━━━━━━━┫ error: bad call\n"
         "       · ╭────────╯       │\n"
         "       · │ `ExtremelyLongUnbreakableTypeNameForTestingNumberOne`\n"
         "       · ╭────────────────╯\n"
         "       · │ `ExtremelyLongUnbreakableTypeNameForTestingNumberTwo`\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ALabelFollowsAnOutDentedBlock) {
  // An out-dented block takes its place in reading order like any label row,
  // so a later label is drawn below the block. Its label line skips the block's
  // text rows, and its text still starts right of every label line, including
  // an out-dented label's.
  static constexpr llvm::StringLiteral Line = "  Run0(first, second);";
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = Line,
               .line_number = 4,
               .column_number = column,
               .length = length};
  };
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 60};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(3, 4), "bad call"),
      .labels = {
          MakeLabel(LabelCategory::Info, at(8, 5),
                    "`ExtremelyLongUnbreakableTypeNameForTestingNumberOne`"),
          MakeLabel(LabelCategory::Info, at(15, 6), "short")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       ╭─┤ foo.carbon:4:3\n"
         "->   4 │   Run0(first, second);\n"
         "       ·   ━━┳━ ──┬──  ───┬──\n"
         "       ·     ┗━━━━━━━━━━━━━━━┫ error: bad call\n"
         "       · ╭────────╯       │\n"
         "       · │ `ExtremelyLongUnbreakableTypeNameForTestingNumberOne`\n"
         "       ·                  ╰────┤ short\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ARoutedLabelUnderAscii) {
  // The route degrades to the same corner and bar glyphs as the rest of the
  // frame, with the segments running into them telling the corners apart.
  EXPECT_THAT(
      RenderLabel(76,
                  "type `str` does not implement interface `Core.ImplicitAs("
                  "ExtremelyLongNamedGenericContainerTypeForTesting(i32))`",
                  Terminal::Charset::Ascii),
      Eq("       .-| foo.carbon:2:11\n"
         "->   2 |   NeedsIt(\"definitely not the right type\");\n"
         "       :           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
         "       :                          '--| error: bad\n"
         "       : .------------------------'\n"
         "       : | type `str` does not implement interface\n"
         "       : | "
         "`Core.ImplicitAs(ExtremelyLongNamedGenericContainerTypeForTesting("
         "i32))`\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, NarrowWidthFallsBackToCompact) {
  // A width too narrow for a frame gets the compact form whether it came from a
  // terminal or from `COLUMNS` over a pipe. What decides is the width, not what
  // is on the other end of the stream.
  for (bool is_terminal : {false, true}) {
    Terminal::Capabilities capabilities = {.is_terminal = is_terminal,
                                           .columns = 40};
    EXPECT_THAT(RenderOne(capabilities, Level::Error, At(4, 3, 7), "bad call"),
                Eq("foo.carbon:4:3-9: error: bad call\n"))
        << is_terminal;
  }
}

TEST(RendererTest, InvalidUtf8IsReplacedUnderUtf8) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Loc loc = {.filename = "foo.carbon",
             .line = "x = \xff\xfe;",
             .line_number = 1,
             .column_number = 1,
             .length = 1};
  // The bytes have no rendering of their own, so the buffer replaces them, and
  // each still takes the one column the underline was measured against.
  EXPECT_THAT(RenderOne(capabilities, Level::Error, loc, "bad"),
              Eq("       ╭─┤ foo.carbon:1:1\n"
                 "->   1 │ x = ��;\n"
                 "       · ┳\n"
                 "       · ┗━━┫ error: bad\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ARangeStartingInsideAWideCharacterCoversItWhole) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  // Clang's columns are raw byte offsets, so one can land inside a symbol.
  // Splitting it would measure the bytes before the split as loose bytes, and
  // underline a fraction of a character that takes two columns to draw.
  Loc loc = {.filename = "foo.carbon",
             .line =
                 "a\xe4\xb8\x96"
                 "b",
             .line_number = 1,
             .column_number = 3,
             .length = 1};
  EXPECT_THAT(RenderOne(capabilities, Level::Error, loc, "bad"),
              Eq("       ╭─┤ foo.carbon:1:3\n"
                 "->   1 │ a世b\n"
                 "       ·  ━┳\n"
                 "       ·   ┗━━┫ error: bad\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, CarriageReturnIsNotDrawn) {
  Loc loc = {.filename = "foo.carbon",
             .line = "var x;\r",
             .line_number = 1,
             .column_number = 5,
             .length = 1};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("       .-| foo.carbon:1:5\n"
                 "->   1 | var x;\n"
                 "       :     ^\n"
                 "       :     '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, PrimaryLabelUnderUtf8) {
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "too many arguments"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), "here")}};
  // A primary range turns into its label the same way an informational one
  // does; only the underline itself differs.
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ━━━┳━━━\n"
                 "       ·      ┗━━┫ error: too many arguments\n"
                 "       ·      ╰────┤ here\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, LocationInfoAloneFallsBackToCompact) {
  Loc import_loc = {.filename = "c.carbon",
                    .line = "import library \"a\";",
                    .line_number = 1,
                    .column_number = 1,
                    .length = 6};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage({.filename = "b.carbon"}, "bad",
                             {MakeOrigin(import_loc, "in import")})};
  // The only message with source is location information, which never gets a
  // snippet, so there is nothing to frame.
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("in import: c.carbon:1:1\n"
                                              "b.carbon: error: bad\n"));
}

TEST(RendererTest, SnippetsOffSpellsEachRangeInItsLocation) {
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message =
          MakeMessage(At(4, 3, 7), "1 argument passed to function expecting 0"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7),
                           "1 argument passed here")}};
  // With snippets off there is no source to draw an extent under, so the
  // location carries it: the byte column the range starts at through the byte
  // column it ends on.
  EXPECT_THAT(
      Render(Plain(), diagnostic, /*snippets=*/false),
      Eq("foo.carbon:4:3-9: error: 1 argument passed to function expecting 0\n"
         "foo.carbon:4:3-9: note: 1 argument passed here\n"
         ""));
}

TEST(RendererTest, AWordlessLabelSaysNothingInTheCompactForm) {
  // The label is attached to a range, and the compact form has no source to
  // underline it in, so it contributes no line rather than an empty one.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), "")}};
  EXPECT_THAT(Render(Narrow(), diagnostic),
              Eq("foo.carbon:4:3: error: bad call\n"));
}

TEST(RendererTest, ARepeatedPathIsDrawnOnce) {
  // A context and the message reached the same way share one drawn path.
  Loc import_loc = {
      .filename = "c.carbon", .line_number = 1, .column_number = 1};
  llvm::SmallVector<LocationInfo, 0> origin = {
      MakeOrigin(import_loc, "imported from")};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad call", origin),
      .contexts = {MakeContext(At(4, 3, 1), "in call", origin)}};
  EXPECT_THAT(Render(Narrow(), diagnostic),
              Eq("imported from: c.carbon:1:1\n"
                 "foo.carbon:4:3: error: in call\n"
                 "foo.carbon:4:3-9: note: bad call\n"));
}

TEST(RendererTest, SnippetsOffPutsAContextLocationAfterItsWords) {
  Loc import_loc = {.filename = "a.carbon",
                    .line_number = 2,
                    .column_number = 1,
                    .length = 6};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "bad",
                             {MakeOrigin(import_loc, "in import")})};
  EXPECT_THAT(Render(Plain(), diagnostic, /*snippets=*/false),
              Eq("in import: a.carbon:2:1\n"
                 "foo.carbon:4:3-9: error: bad\n"));
}

TEST(RendererTest, SnippetsOffRendersOneRowPerPart) {
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 7), "too many arguments"),
      .labels = {
          MakeLabel(LabelCategory::Info, At(1, 1, 12), "declared here")}};
  EXPECT_THAT(Render(Plain(), diagnostic, /*snippets=*/false),
              Eq("foo.carbon:4:3-9: error: too many arguments\n"
                 "foo.carbon:1:1-12: note: declared here\n"));
}

TEST(RendererTest, PrintSnippetIndents) {
  RawStringOstream out;
  PrintSnippet(out, At(4, 3, 7), /*indent=*/10);
  EXPECT_THAT(out.TakeStr(), Eq("          4 |   Run0(1);\n"
                                "            :   -------\n"
                                ""));
}

TEST(RendererTest, PrintSnippetWithNoColumnPrintsNothing) {
  RawStringOstream out;
  PrintSnippet(out, {.filename = "foo.carbon", .line_number = 3},
               /*indent=*/0);
  EXPECT_THAT(out.TakeStr(), Eq(""));
}

TEST(RendererTest, FormatLocationDropsUnknownParts) {
  EXPECT_THAT(FormatLocation({}), Eq(""));
  EXPECT_THAT(FormatLocation({.filename = "a.carbon"}), Eq("a.carbon"));
  EXPECT_THAT(FormatLocation({.filename = "a.carbon", .line_number = 3}),
              Eq("a.carbon:3"));
  EXPECT_THAT(
      FormatLocation(
          {.filename = "a.carbon", .line_number = 3, .column_number = 4}),
      Eq("a.carbon:3:4"));
}

TEST(RendererTest, AWidthPastWhatAGridHoldsIsStillRendered) {
  // `COLUMNS` is whatever a user exported, so a width past what a buffer can
  // hold is bad input rather than a caller mistake, in the compact form as
  // much as in the frame.
  for (int columns : {1 << 20, std::numeric_limits<int>::max()}) {
    Terminal::Capabilities capabilities = {.is_terminal = true,
                                           .columns = columns};
    EXPECT_THAT(RenderOne(capabilities, Level::Error, At(4, 3, 7), "wide"),
                HasSubstr("Run0(1);"))
        << columns;
    EXPECT_THAT(Render(capabilities,
                       {.level = Level::Error,
                        .message = MakeMessage(At(4, 3, 7), "wide")},
                       /*snippets=*/false),
                HasSubstr("wide"))
        << columns;
  }
}

TEST(RendererTest, TheShapeTheHeaderDraws) {
  // The picture in `renderer.h` is what this produces, rather than one drawn by
  // hand beside it. A rendering that stops matching it is a rendering whose
  // documentation has gone stale.
  Terminal::Capabilities capabilities = {.charset = Terminal::Charset::Utf8};
  std::string file = "fn Run0() {}\n" + std::string(17, '\n') + "  Run0(1);\n";
  Loc decl = {.filename = "foo.carbon",
              .line = "fn Run0() {}",
              .file_text = file,
              .line_number = 1,
              .column_number = 1,
              .length = 12};
  Loc call = {.filename = "foo.carbon",
              .line = "  Run0(1);",
              .file_text = file,
              .line_number = 19,
              .column_number = 3,
              .length = 7};
  EXPECT_THAT(
      Render(
          capabilities,
          {.level = Level::Error,
           .message = MakeMessage(call, "invalid arguments in function call"),
           .labels = {MakeLabel(LabelCategory::Info, decl,
                                "calling function declared here")}}),
      Eq("       ╭─┤ foo.carbon:19:3\n"
         "     1 │ fn Run0() {}\n"
         "       · ──────┬─────\n"
         "       ·       ╰──┤ calling function declared here\n"
         "       ┆\n"
         "->  19 │   Run0(1);\n"
         "       ·   ━━━┳━━━\n"
         "       ·      ┗━━┫ error: invalid arguments in function call\n"
         "       ╰────\n"
         ""));
}

TEST(RendererTest, ARangeOnAnotherLineDoesNotStandInForTheLocation) {
  // Only a range on the line the message's location names stands in for it. One
  // somewhere else says nothing about that line, and taking its place would
  // hang the message's words off code they aren't about while dropping the line
  // they are, so the message keeps its own underline and both lines show.
  EXPECT_THAT(Render(Plain(), {.level = Level::Error,
                               .message = MakeMessage(At(1, 1, 2), "bad"),
                               .labels = {MakeLabel(LabelCategory::Primary,
                                                    At(4, 3, 7), "here")}}),
              Eq("       .-| foo.carbon:1:1\n"
                 "->   1 | fn Run0() {}\n"
                 "       : ^^\n"
                 "       :  '--| error: bad\n"
                 "       :\n"
                 "     4 |   Run0(1);\n"
                 "       :   ---.---\n"
                 "       :      '--| here\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, RotateLeavesTheHeavyUnderlineTheLevelColor) {
  // The level word, the `->` in the margin, the reported line's number, and
  // the heavy underline identify the diagnostic, so they keep the level's
  // center color. The other problem ranges take colors from the ramp with the
  // center left out, starting from the dark end.
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Truecolor};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .contexts = {MakeContext(At(4, 3, 4), "in call")},
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 8, 1), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  EXPECT_THAT(rendered, HasSubstr("\x1b[1m\x1b[38;2;255;114;97merror: "));
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;255;114;97m->"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[1m\x1b[38;2;255;114;97m^^"));
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;181;86;0m-"));
}

TEST(RendererTest, ARampRunsOutAndRepeats) {
  // A named-color theme is a ramp of two, so a third range of one theme wraps
  // back to the ramp's start rather than reaching outside its theme.
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi16};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(LineOnly(4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Info, At(4, 3, 2), ""),
                 MakeLabel(LabelCategory::Info, At(4, 6, 2), ""),
                 MakeLabel(LabelCategory::Info, At(4, 9, 2), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  // Plain cyan starts the ramp and the third range comes back around to it;
  // bright cyan is the second.
  auto count = [&](llvm::StringRef needle) {
    int found = 0;
    for (size_t at = llvm::StringRef(rendered).find(needle);
         at != llvm::StringRef::npos;
         at = llvm::StringRef(rendered).find(needle, at + 1)) {
      ++found;
    }
    return found;
  };
  EXPECT_THAT(count("\x1b[36m"), Eq(2)) << rendered;
  EXPECT_THAT(count("\x1b[96m"), Eq(1)) << rendered;
}

TEST(RendererTest, Ansi256GetsTheChosenColors) {
  // `Ansi256` renders the chosen palettes rounded to its cube rather than the
  // sixteen named colors.
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Ansi256};
  std::string rendered =
      RenderOne(capabilities, Level::Error, At(4, 3, 7), "bad call");
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;5;"));
}

TEST(RendererTest, AHostedMessagesLabelLineFollowsTheHostsRampColor) {
  // With a context leading, the message's words hang from the range containing
  // its location, and the rotation recolors that host: the label line and bar
  // follow the host's rotated color rather than keeping the theme's center.
  Terminal::Capabilities capabilities = {.color_mode =
                                             Terminal::ColorMode::Truecolor};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 4, 1), "bad call"),
      .contexts = {MakeContext(At(3, 4, 4), "in `Main`")},
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 4), ""),
                 MakeLabel(LabelCategory::Primary, At(4, 8, 2), "")}};
  std::string rendered = Render(capabilities, diagnostic);
  // The host is the leftmost problem range on its line and takes the dark
  // end; the message's label line reaches its label in the same color.
  EXPECT_THAT(rendered, HasSubstr("\x1b[38;2;181;86;0m'--|")) << rendered;
}

TEST(RendererTest, ASpanPastTheEndOfAFullWidthLineIsWindowedIn) {
  // An 80-column line exactly fills the unstated-width target, and the span
  // names the column after its last character -- a missing `;`. The span is
  // part of the width, so the line is windowed and the underline lands inside
  // the columns rather than one past every cell there is.
  std::string line(80, 'x');
  Loc loc = {.filename = "foo.carbon",
             .line = line,
             .line_number = 1,
             .column_number = 81,
             .length = 1};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "missing `;`"),
              Eq("       .-| foo.carbon:1:81\n"
                 "->   1 | "
                 "..."
                 "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                 "xxxxxxxxxxxx\n"
                 "       :                                                     "
                 "                        ^\n"
                 "       : "
                 ".------------------------------------------------------------"
                 "---------------'\n"
                 "       : | error: missing `;`\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ASpanOutsideASharedWindowIsClampedToIt) {
  // Spans sharing a row share the first span's window; one lying past that
  // window's right edge is clamped to its last column rather than drawn past
  // the buffer.
  std::string line(300, 'x');
  line.replace(30, 5, "FIRST");
  line.replace(200, 3, "FAR");
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = line,
               .line_number = 1,
               .column_number = column,
               .length = length};
  };
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(33, 1), "bad"),
      .labels = {MakeLabel(LabelCategory::Primary, at(31, 5), ""),
                 MakeLabel(LabelCategory::Primary, at(201, 3), "")}};
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       .-| foo.carbon:1:33\n"
         "->   1 | "
         "...xxxxxxxxxxxxxxxxFIRSTxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx."
         "..\n"
         "       :                    ^^^^^                                    "
         "          -\n"
         "       :                      '--| error: bad\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, ASpanCutOffTheLeftOfAWindowIsNotCarried) {
  // A span windowed out to the left is clamped rather than keeping its full
  // length and underlining the unrelated source that took its place at the
  // window's start. The message's location, at the far-right span, focuses the
  // shared window there; the length-40 label at column 11 falls off the left.
  std::string line(300, 'x');
  auto at = [&](int column, int length) {
    return Loc{.filename = "foo.carbon",
               .line = line,
               .line_number = 1,
               .column_number = column,
               .length = length};
  };
  Terminal::Capabilities capabilities = {.is_terminal = true, .columns = 80};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at(252, 1), "bad"),
      .labels = {MakeLabel(LabelCategory::Primary, at(250, 5), ""),
                 MakeLabel(LabelCategory::Primary, at(11, 40), "")}};
  // The window shows the right span; the left label, cut off, is a one-column
  // underline on the leading `...` rather than dragging its 40-column length
  // across the source that took its place.
  EXPECT_THAT(
      Render(capabilities, diagnostic),
      Eq("       .-| foo.carbon:1:252\n"
         "->   1 | "
         "...xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx."
         "..\n"
         "       : -                  ^^^^^\n"
         "       :                      '--| error: bad\n"
         "       '----\n"
         ""));
}

TEST(RendererTest, AControlByteInAFilenameCannotBreakTheFrame) {
  // A filename is drawn on one row like anything else, so a newline or other
  // control byte in it is replaced rather than moving the cursor off the row.
  Loc loc = {.filename = "ab\ncd\ttab.carbon",
             .line = "var x: i32",
             .line_number = 1,
             .column_number = 5,
             .length = 1};
  std::string rendered = RenderOne(Plain(), Level::Error, loc, "bad");
  EXPECT_THAT(rendered, HasSubstr("ab cd tab.carbon:1:5")) << rendered;
  // The location row stays a single row: the location's own bytes add no
  // newline of their own, so the frame still has exactly the rows it draws.
  EXPECT_THAT(rendered, Not(HasSubstr("\n\n"))) << rendered;
}

TEST(RendererTest, AMergedPairWithoutFileTextStillElides) {
  // A single skipped line is shown in place of an elision row, but only when
  // `Loc::file_text` offers a way to find it; without it the elision row says
  // a line was skipped rather than a blank row claiming the line is empty.
  auto at = [](llvm::StringRef line_text, int line_number) {
    return Loc{.filename = "foo.carbon",
               .line = line_text,
               .line_number = line_number,
               .column_number = 1,
               .length = 2};
  };
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(at("first line", 1), "bad"),
      .labels = {MakeLabel(LabelCategory::Info, at("third line", 3), "here")}};
  EXPECT_THAT(Render(Plain(), diagnostic), Eq("       .-| foo.carbon:1:1\n"
                                              "->   1 | first line\n"
                                              "       : ^^\n"
                                              "       :  '--| error: bad\n"
                                              "       :\n"
                                              "     3 | third line\n"
                                              "       : -.\n"
                                              "       :  '--| here\n"
                                              "       '----\n"
                                              ""));
}

TEST(RendererTest, ControlCharactersInWordsBecomeSpaces) {
  // Every row is one row: a newline in a message's text would otherwise be
  // measured as one column and drawn over the row below.
  EXPECT_THAT(RenderOne(Plain(), Level::Error, At(4, 3, 7), "bad\ncall"),
              HasSubstr("error: bad call\n"));
}

TEST(RendererTest, AWhitespaceOnlyLabelIsJustAnUnderline) {
  // A label whose text scrubs away to spaces has nothing to hang: it becomes
  // an underline with no label line, rather than rows framing nothing.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 4), "bad call"),
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 8, 2), "\n")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:4:3\n"
                 "->   4 |   Run0(1);\n"
                 "       :   ^^^^ --\n"
                 "       :     '--| error: bad call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AZeroLengthSpanIsDrawnAsAPoint) {
  // A `Loc` promises a length of at least 1; one of 0 -- an insertion point
  // from a fix-it -- is drawn as a single column rather than trusted.
  Loc loc = {.filename = "foo.carbon",
             .line = "var x: i32",
             .line_number = 1,
             .column_number = 5,
             .length = 0};
  EXPECT_THAT(RenderOne(Plain(), Level::Error, loc, "bad"),
              Eq("       .-| foo.carbon:1:5\n"
                 "->   1 | var x: i32\n"
                 "       :     ^\n"
                 "       :     '--| error: bad\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, AContextLeadsTheFrameAndTheMessageHangs) {
  // A context with a location leads the diagnostic: the level word and the
  // heavy underline are its, and the message's text is placed like a label's
  // under the range containing its location, in the problem's color.
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 4, 1), "no matching call"),
      .contexts = {MakeContext(At(3, 4, 4), "in `Main`")},
      .labels = {MakeLabel(LabelCategory::Primary, At(4, 3, 7), "")}};
  EXPECT_THAT(Render(Plain(), diagnostic),
              Eq("       .-| foo.carbon:3:4\n"
                 "->   3 | fn Main() {\n"
                 "       :    ^^^^\n"
                 "       :      '--| error: in `Main`\n"
                 "     4 |   Run0(1);\n"
                 "       :   ---.---\n"
                 "       :      '--| no matching call\n"
                 "       '----\n"
                 ""));
}

TEST(RendererTest, ALabelLinePassingAnOutDentedBlockSkipsItsRowsWhole) {
  // An out-dented block occupies its rows from the bar to the right edge,
  // blank cells included, so a label line descending past it skips every row of
  // the block. A line in the indent under the level word would look like a
  // second bar framing the continuation.
  auto routed_at = [](int column, int length) {
    Loc loc = RoutedLoc();
    loc.column_number = column;
    loc.length = length;
    return loc;
  };
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 76};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(
          RoutedLoc(),
          "type `str` does not implement interface `Core.ImplicitAs("
          "ExtremelyLongNamedGenericContainerTypeForTesting(i32))` at all"),
      .labels = {MakeLabel(LabelCategory::Info, routed_at(3, 7), "callee")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:2:11\n"
                 "->   2 │   NeedsIt(\"definitely not the right type\");\n"
                 "       ·   ───┬─── ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━\n"
                 "       · ┏━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
                 "       · ┃ error: type `str` does not implement interface\n"
                 "       · ┃        `Core.ImplicitAs("
                 "ExtremelyLongNamedGenericContainerTypeForTesting(i32))`\n"
                 "       · ┃        at all\n"
                 "       ·      ╰──────────────────────┤ callee\n"
                 "       ╰────\n"
                 ""));
}

TEST(RendererTest, ALabelLineSkipsARoutedLabelsRows) {
  // A routed label's text spans the full width, so the label line of a later
  // label skips its rows and resumes below them.
  Terminal::Capabilities capabilities = {
      .charset = Terminal::Charset::Utf8, .is_terminal = true, .columns = 80};
  Diagnostic diagnostic = {
      .level = Level::Error,
      .message = MakeMessage(At(4, 3, 1), "bad call"),
      .labels = {
          MakeLabel(LabelCategory::Info, At(4, 3, 1), "left words"),
          MakeLabel(LabelCategory::Info, At(4, 8, 2),
                    "`AnExtremelyLongUnbreakableGenericTypeNameForTesting("
                    "AnExtremelyLongUnbreakableGenericTypeNameForTesting)`"),
          MakeLabel(LabelCategory::Info, At(4, 10, 1), "right words")}};
  EXPECT_THAT(Render(capabilities, diagnostic),
              Eq("       ╭─┤ foo.carbon:4:3\n"
                 "->   4 │   Run0(1);\n"
                 "       ·   ┳    ┬╴╷\n"
                 "       ·   ┗━━━━━━━━━┫ error: bad call\n"
                 "       ·   ╰───────────┤ left words\n"
                 "       · ╭──────╯ │\n"
                 "       · │ "
                 "`AnExtremelyLongUnbreakableGenericTypeNameForTesting("
                 "AnExtremelyLongUnbreakableGenericTypeNameForTesting)`\n"
                 "       ·          ╰──────┤ right words\n"
                 "       ╰────\n"
                 ""));
}

}  // namespace
}  // namespace Carbon::Diagnostics

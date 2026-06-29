// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/format.h"
#include "toolchain/format/style.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Format {
namespace {

// Inputs exercised by the invariant tests below. They deliberately include
// messy whitespace, blank lines, comments, and nesting.
constexpr llvm::StringLiteral Inputs[] = {
    "fn F() {}",
    "fn  F ( x : i32 ) -> i32 { return x ; }",
    "class C { var x: i32; }",
    "class C { class D { class E {} } }",
    "fn F() {\n\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n\n}",
    "// A comment\nfn F() {}\n",
    // Member access, designators, and unary operators.
    "fn F(c: C*) -> i32 { return c->x + F2( - c . y ) * a[i]; }",
    "fn F(color: Color) -> i32 { match (color) { case .Red => { return 1; } "
    "default => { return 0; } } }",
    "var v: auto = { .a = 1,.b = 2 };",
    // Lines long enough to force the wrapping solver to break them, so the
    // invariants below also cover wrapped output.
    "fn RegisterHandler(event_name: String, priority: i32, callback: Callback)"
    " -> bool { return true; }",
    "fn F() {\n  RegisterHandler(event_name, default_priority, my_callback,"
    " extra_argument_value);\n}",
    "fn F() {\n  var is_valid: bool = has_permission and is_authenticated and"
    " not is_token_expired;\n}",
    // An operator chain long enough to wrap, exercising operand alignment.
    "fn F() {\n  return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd +"
    " eeeeeeeeee * ffffffffff;\n}",
    // Comments that need re-indentation and wrapping.
    "fn F() {\n        // over-indented comment\n  // a very long comment line"
    " that runs well past the eighty column limit and therefore has to wrap"
    " around;\n  var x: i32 = 1;\n}",
    // Trailing comments kept on their code lines.
    "var x: i32 = 1; // a\nvar yyyy: i32 = 22; // b\n",
    // A member-access chain long enough to wrap before a `.`.
    "fn F() {\n  receiverrrrrrrrrrr.aaaaaaaaaaaaaaaaa().bbbbbbbbbbbbbbbbb()"
    ".cccccccccccccccc();\n}",
    // Overflowing declarations that must break before `->` or after a binding
    // colon rather than splitting at a keyword.
    "class C { private fn Configure() -> "
    "ValidationResultStatusNameGoesHereAndOverflowsXyzzy; }",
    "fn T() {\n  var accumulated_statistics_record_value: "
    "ProcessingStatisticsRecordContainerXXXXX;\n}",
    // Overflowing lines whose only column-saving break would put whitespace
    // inside a unary operator; they must stay overlong instead.
    "fn U() {\n  TakeTheAddress("
    "&some_rather_long_variable_name_that_makes_this_overflow_xyzzy);\n}",
    "fn R() {\n  TakePointerType("
    "SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);\n}",
};

// Formats `text` and returns the result.
auto FormatText(Testing::CompileHelper& helper, llvm::StringRef text)
    -> std::string {
  RawStringOstream out;
  Format(helper.GetTree(text), out);
  return out.TakeStr();
}

// Returns the newline-separated sequence of token kinds in `text`, for
// comparing the tokens of two sources.
auto TokenKinds(Testing::CompileHelper& helper, llvm::StringRef text)
    -> std::string {
  Lex::TokenizedBuffer& tokens = helper.GetTokenizedBuffer(text);
  RawStringOstream out;
  for (auto token : tokens.tokens()) {
    out << tokens.GetKind(token).name() << "\n";
  }
  return out.TakeStr();
}

// Formatting formatted output must be a no-op.
TEST(FormatterTest, Idempotent) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    std::string once = FormatText(helper, input);
    std::string twice = FormatText(helper, once);
    EXPECT_EQ(once, twice) << "not idempotent for input:\n" << input;
  }
}

// Error-free input formats to error-free output: no layout decision may
// introduce an error, such as a break that puts whitespace inside a unary
// operator, which the parser's fixity rules reject.
TEST(FormatterTest, OutputStaysErrorFree) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    if (helper.GetTree(input).has_errors()) {
      continue;
    }
    std::string formatted = FormatText(helper, input);
    EXPECT_FALSE(helper.GetTree(formatted).has_errors())
        << "formatting introduced errors for input:\n"
        << input << "\noutput:\n"
        << formatted;
  }
}

// A non-empty file always ends with exactly one newline, even when the source
// ends mid-line -- including in a comment, whose text otherwise carries its own
// trailing newline.
TEST(FormatterTest, EndsWithSingleNewline) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : {"var x: i32 = 1;", "// only a comment"}) {
    std::string formatted = FormatText(helper, input);
    EXPECT_TRUE(llvm::StringRef(formatted).ends_with("\n"))
        << "missing final newline for input:\n"
        << input;
    EXPECT_FALSE(llvm::StringRef(formatted).ends_with("\n\n"))
        << "extra final newline for input:\n"
        << input;
  }
}

// Formatting only changes whitespace, so re-lexing the output must yield the
// same token sequence as the input.
TEST(FormatterTest, PreservesTokens) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    std::string formatted = FormatText(helper, input);
    EXPECT_EQ(TokenKinds(helper, input), TokenKinds(helper, formatted))
        << "tokens changed for input:\n"
        << input;
  }
}

// A trailing comment is kept on the line of the code it follows, separated by a
// single space (a stray double space is normalized). Lining up runs of trailing
// comments into a column comes later, with the whitespace manager.
TEST(FormatterTest, TrailingCommentStaysOnCodeLine) {
  Testing::CompileHelper helper;
  EXPECT_EQ(FormatText(helper, "var x: i32 = 0;// c\n"),
            "var x: i32 = 0; // c\n");
}

// Formats `text` via `FormatReplacements` + `ApplyReplacements`.
auto FormatViaReplacements(Testing::CompileHelper& helper, llvm::StringRef text,
                           std::optional<LineRange> lines = std::nullopt)
    -> std::string {
  llvm::SmallVector<Replacement> replacements;
  FormatReplacements(helper.GetTree(text), replacements, lines);
  return ApplyReplacements(text, replacements);
}

// Applying the whole-document replacements must reproduce `Format`'s output.
TEST(FormatterTest, ReplacementsReproduceFormat) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    EXPECT_EQ(FormatViaReplacements(helper, input), FormatText(helper, input))
        << "replacements diverged from Format for input:\n"
        << input;
  }
}

// The replacement invariant holds even for lex-broken input whose recovery
// inserts tokens: a recovery token's text exists in no source byte range, so it
// is not an edit anchor and must flow into a neighboring gap's edit instead.
// The two inputs cover a single recovery token and two at the same offset.
TEST(FormatterTest, ReplacementsReproduceFormatOnRecoveredInput) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : {"class C { fn F( }", "class C { fn F( [ }"}) {
    EXPECT_EQ(FormatViaReplacements(helper, input), FormatText(helper, input))
        << "replacements diverged from Format for input:\n"
        << input;
  }
}

// Already-formatted input produces no edits at all.
TEST(FormatterTest, NoReplacementsWhenAlreadyFormatted) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    std::string formatted = FormatText(helper, input);
    llvm::SmallVector<Replacement> replacements;
    FormatReplacements(helper.GetTree(formatted), replacements);
    EXPECT_TRUE(replacements.empty()) << "expected no edits re-formatting:\n"
                                      << formatted;
  }
}

// A single localized change yields a single small edit, not a whole-file
// rewrite.
TEST(FormatterTest, ProducesMinimalEdits) {
  Testing::CompileHelper helper;
  llvm::SmallVector<Replacement> replacements;
  // Only the spacing around `+` on the middle line is wrong (the input is
  // otherwise already formatted, including its trailing newline).
  FormatReplacements(
      helper.GetTree("fn F() {\n  var x: i32 = 1+2;\n  return x;\n}\n"),
      replacements);
  ASSERT_EQ(replacements.size(), 2);
  for (const Replacement& replacement : replacements) {
    EXPECT_EQ(replacement.text, " ");
  }
}

// A line range restricts edits to that range, leaving other lines untouched.
TEST(FormatterTest, RangeFormattingTouchesOnlyTheRange) {
  Testing::CompileHelper helper;
  llvm::StringRef input =
      "fn F() {\n  var x: i32 = 1+1;\n  var y: i32 = 2+2;\n}\n";
  // Line 2 is reformatted; line 3 keeps its `2+2`.
  EXPECT_EQ(FormatViaReplacements(helper, input,
                                  LineRange{.first_line = 2, .last_line = 2}),
            "fn F() {\n  var x: i32 = 1 + 1;\n  var y: i32 = 2+2;\n}\n");
  // The complementary range reformats line 3 only.
  EXPECT_EQ(FormatViaReplacements(helper, input,
                                  LineRange{.first_line = 3, .last_line = 3}),
            "fn F() {\n  var x: i32 = 1+1;\n  var y: i32 = 2 + 2;\n}\n");
}

// Range formatting expands to a matching brace: the over-indented `}` on line 3
// is out of the requested range -- and its own gap starts on line 2, also out
// of range -- but its matching `{` (line 1) is in range, so it is fixed too.
TEST(FormatterTest, RangeFormattingExpandsToMatchingBrace) {
  Testing::CompileHelper helper;
  EXPECT_EQ(
      FormatViaReplacements(helper, "fn F() {\n  var x: i32 = 1;\n      }\n",
                            LineRange{.first_line = 1, .last_line = 1}),
      "fn F() {\n  var x: i32 = 1;\n}\n");
}

// Range formatting expands to whole unwrapped lines: a statement wrapped over
// source lines 2-3 re-wraps as a unit even when only line 2 is requested, so
// the applied edits match what full formatting would produce for it.
TEST(FormatterTest, RangeFormattingRewrapsWholeUnwrappedLine) {
  Testing::CompileHelper helper;
  EXPECT_EQ(
      FormatViaReplacements(helper, "fn F() {\n  var x: i32 =\n      1+1;\n}\n",
                            LineRange{.first_line = 2, .last_line = 2}),
      "fn F() {\n  var x: i32 = 1 + 1;\n}\n");
}

// Formats `text` with a specific style.
auto FormatTextWithStyle(Testing::CompileHelper& helper, llvm::StringRef text,
                         const Style& style) -> std::string {
  RawStringOstream out;
  Format(helper.GetTree(text), out, style);
  return out.TakeStr();
}

// `indent_width` controls how far each brace level indents.
TEST(FormatterTest, StyleIndentWidth) {
  Testing::CompileHelper helper;
  Style style;
  style.indent_width = 4;
  EXPECT_EQ(FormatTextWithStyle(helper, "fn F() {\n  return x;\n}\n", style),
            "fn F() {\n    return x;\n}\n");
}

// A narrower `column_limit` wraps a line that fits at the default limit.
TEST(FormatterTest, StyleColumnLimitControlsWrapping) {
  Testing::CompileHelper helper;
  llvm::StringRef input = "fn F() {\n  foo(aaaa, bbbb, cccc);\n}\n";
  Style narrow;
  narrow.column_limit = 20;
  std::string wide = FormatTextWithStyle(helper, input, Style());
  std::string tight = FormatTextWithStyle(helper, input, narrow);
  EXPECT_LT(llvm::count(wide, '\n'), llvm::count(tight, '\n'))
      << "narrow limit did not add line breaks; tight output:\n"
      << tight;
}

// `max_empty_lines_to_keep` bounds the blank lines kept between statements.
TEST(FormatterTest, StyleMaxEmptyLinesToKeep) {
  Testing::CompileHelper helper;
  llvm::StringRef input =
      "fn F() {\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n}\n";
  EXPECT_EQ(FormatTextWithStyle(helper, input, Style()),
            "fn F() {\n  var x: i32 = 1;\n\n  var y: i32 = 2;\n}\n");
  Style keep_two;
  keep_two.max_empty_lines_to_keep = 2;
  EXPECT_EQ(FormatTextWithStyle(helper, input, keep_two),
            "fn F() {\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n}\n");
}

// A run of trailing comments aligns into one column under the canonical style.
TEST(FormatterTest, TrailingCommentsAlignedByDefault) {
  Testing::CompileHelper helper;
  EXPECT_EQ(
      FormatTextWithStyle(
          helper, "var x: i32 = 1; // a\nvar yyyy: i32 = 22; // b\n", Style()),
      "var x: i32 = 1;     // a\nvar yyyy: i32 = 22; // b\n");
}

// A blank line breaks a trailing-comment run, so each comment keeps its single
// space rather than aligning across the gap.
TEST(FormatterTest, TrailingCommentAlignmentStopsAtBlankLine) {
  Testing::CompileHelper helper;
  llvm::StringRef input = "var x: i32 = 1; // a\n\nvar yyyy: i32 = 22; // b\n";
  EXPECT_EQ(FormatTextWithStyle(helper, input, Style()), input);
}

// `align_trailing_comments` can be turned off; the comments then keep a single
// space and are not lined up.
TEST(FormatterTest, StyleAlignTrailingCommentsCanBeDisabled) {
  Testing::CompileHelper helper;
  llvm::StringRef input = "var x: i32 = 1; // a\nvar yyyy: i32 = 22; // b\n";
  Style style;
  style.align_trailing_comments = false;
  EXPECT_EQ(FormatTextWithStyle(helper, input, style), input);
}

}  // namespace
}  // namespace Carbon::Format

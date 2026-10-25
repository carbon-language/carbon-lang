// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <string>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/format/format.h"
#include "toolchain/format/style.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Format {
namespace {

constexpr llvm::StringLiteral Inputs[] = {
    "fn  F ( x : i32 ) -> i32 { return x ; }",
    "namespace A . B ; fn Vector ( T : type ) . Get [ self : Self ] ( ) { }",
    "class C { class D { class E {} } }",
    "fn F() {\n\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n\n}",
    "// only a comment",
    "fn F(c: C*, t: ((i32, i32), i32)) -> i32 { return c->x + F2( -c . y ) * "
    "a[i] + t . 0 . 1 + 2 .abs; }",
    "fn F(p: {}*) -> {.x: i32} { var v: auto = { .a = 1, .b = 2, }; return "
    "{.x = {.a = 1}.a}; }",
    "fn F(color: Color) -> i32 { match (color) { case .Red => { return 1; } "
    "case {.x = a: i32} => { return a; } default => { return 0; } } }",
    "choice Color { Red, // first\n Green(x: i32, y: i32), // second\n Blue }",
    "fn F() -> auto { return #\"\"\"// \"#; }",
    "fn RegisterHandler(event_name: String, priority: i32, callback: Callback)"
    " -> bool { return true; }",
    "fn F() {\n  RegisterHandler(event_name, default_priority, my_callback,"
    " extra_argument_value);\n  var is_valid: bool = has_permission and"
    " is_authenticated and not is_token_expired;\n}",
    "fn F() {\n  return {.first_field = alpha_value, .second_field = "
    "beta_value, .third_field = gamma_value};\n}",
    "fn F() {\n  return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd +"
    " eeeeeeeeee * ffffffffff;\n}",
    "fn F() {\n        // over-indented\n  // a very long comment line that"
    " runs well past the eighty column limit and therefore wraps around;\n"
    "  var x: i32 = 1; // a\n  var yyyy: i32 = 22; // b\n}",
    "fn F(a: i32, // c1\n b: i32) -> // ret\n i32 {\n  return a +\n  // add b\n"
    "  b;\n}\n",
    "fn F() {\n  receiverrrrrrrrrrr.aaaaaaaaaaaaaaaaa().bbbbbbbbbbbbbbbbb()"
    ".cccccccccccccccc();\n}",
    "class C { private fn Configure() -> "
    "ValidationResultStatusNameGoesHereAndOverflowsXyzzy; }",
    "fn U() {\n  var accumulated_statistics_record_value: "
    "ProcessingStatisticsRecordContainerXXXXX;\n  TakeTheAddress("
    "&some_rather_long_variable_name_that_makes_this_overflow_xyzzy);\n  "
    "TakePointerType("
    "SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);\n}",
};

auto FormatText(Testing::CompileHelper& helper, llvm::StringRef text,
                const Style& style = Style()) -> std::string {
  RawStringOstream out;
  Format(helper.GetTree(text), out, style);
  return out.TakeStr();
}

auto TokenKinds(Testing::CompileHelper& helper, llvm::StringRef text)
    -> std::string {
  Lex::TokenizedBuffer& tokens = helper.GetTokenizedBuffer(text);
  RawStringOstream out;
  for (auto token : tokens.tokens()) {
    out << tokens.GetKind(token).name() << "\n";
  }
  return out.TakeStr();
}

TEST(FormatterTest, InvariantsHoldAcrossConstructs) {
  Testing::CompileHelper helper;
  for (llvm::StringRef input : Inputs) {
    ASSERT_FALSE(helper.GetTree(input).has_errors()) << input;
    std::string once = FormatText(helper, input);
    EXPECT_TRUE(llvm::StringRef(once).ends_with("\n")) << input;
    EXPECT_FALSE(llvm::StringRef(once).ends_with("\n\n")) << input;
    EXPECT_EQ(FormatText(helper, once), once) << input;
    EXPECT_FALSE(helper.GetTree(once).has_errors()) << once;
    EXPECT_EQ(TokenKinds(helper, input), TokenKinds(helper, once)) << input;
  }
}

TEST(FormatterTest, GreedyFallbackAndLongOperatorChains) {
  Testing::CompileHelper helper;
  // A trailing `callee()` keeps `scope_open_ahead` live across 1,500 arguments
  // so uniform-cost search hits `MaxStatesGenerated` and exercises greedy
  // packing.
  std::string wide_call = "fn F() {\n  Call(";
  for (int i = 0; i < 1500; ++i) {
    wide_call += "elem, ";
  }
  wide_call += "callee());\n}\n";
  std::string packed = FormatText(helper, wide_call);
  EXPECT_GT(llvm::count(packed, '\n'), 100);
  llvm::SmallVector<llvm::StringRef> lines;
  llvm::SplitString(packed, lines, "\n");
  for (llvm::StringRef line : lines) {
    EXPECT_LE(static_cast<int>(line.size()), 80) << line;
  }
  EXPECT_EQ(FormatText(helper, packed), packed);

  // Homogeneous operator chains coalesce identical alignment scopes cleanly.
  std::string chain = "fn F() -> i32 {\n  return ";
  for (int i = 0; i < 50; ++i) {
    if (i > 0) {
      chain += " + ";
    }
    chain += "term_" + std::to_string(i);
  }
  chain += ";\n}\n";
  std::string formatted_chain = FormatText(helper, chain);
  EXPECT_EQ(formatted_chain, FormatText(helper, formatted_chain));

  // Post-lexing recovery identifiers (`x.class`) must not leak operand scopes.
  EXPECT_EQ(
      FormatText(helper,
                 "fn F() {\n  CallFunction((x.class + 1), "
                 "second_rather_long_argument_name_for_wrapping_xyz);\n}"),
      "fn F() {\n  CallFunction((x.class + 1),\n               "
      "second_rather_long_argument_name_for_wrapping_xyz);\n}\n");
}

TEST(FormatterTest, ErrorRecoveryAndNewlineLessLiterals) {
  Testing::CompileHelper helper;
  // Lexer-synthesized recovery brackets are never emitted.
  EXPECT_EQ(FormatText(helper, "fn F() {\n"), "fn F() {\n");
  EXPECT_EQ(FormatText(helper, "var v: auto = {.a = 1"),
            "var v: auto = {.a = 1\n");
  EXPECT_EQ(FormatText(helper, "var v: auto = {\n  .a = 1"),
            "var v: auto = {\n  .a = 1\n");
  // Adjacent symbol tokens (`- -x`) never merge into `--x` even under error
  // recovery.
  EXPECT_EQ(FormatText(helper, "fn F(x: i32) -> i32 { return - -x; }"),
            "fn F(x: i32) -> i32 {\n  return - -x;\n}\n");

  // Mid-statement comment before an error subtree breaks onto a continuation
  // line without corrupting the comment.
  EXPECT_EQ(
      FormatText(helper,
                 "fn F() {\n  var  x : i32  = // explain\n  1  +  +  2 ;\n}\n"),
      "fn F() {\n  var x: i32 = // explain\n      1  +  + 2;\n}\n");

  // Newline-sensitive `#*"""` raw strings and adjacent `""` + `"..."` tokens at
  // EOF stay lex-clean and idempotent when the trailing `\n` is appended.
  struct Case {
    llvm::StringRef input;
    llvm::StringRef expected;
  };
  for (Case c : {
           Case{"var s: auto = ##\"\"\"cpp // \"##;",
                "var s: auto = ##\"\"\\##\"cpp // \"##;\n"},
           Case{"var x: i32 = + #\"\"\"// \"#",
                "var x: i32 = + #\"\"\\#\"// \"#\n"},
           Case{"var x: i32 = + // c\n #\"\"\"// \"#",
                "var x: i32 = + // c\n #\"\"\\#\"// \"#\n"},
           Case{"f{\"\"\"// \"}", "f{\"\" \"// \"}\n"},
       }) {
    std::string formatted = FormatText(helper, c.input);
    EXPECT_EQ(formatted, c.expected);
    EXPECT_FALSE(helper.GetTokenizedBuffer(formatted).has_errors());
    EXPECT_EQ(FormatText(helper, formatted), formatted);
  }
}

TEST(FormatterTest, CrlfAndNonWrappableComments) {
  Testing::CompileHelper helper;
  EXPECT_EQ(FormatText(helper, "// one\r\n  // two\r\nfn F() {}\r\n"),
            "// one\n// two\nfn F() {}\n");

  Style narrow;
  narrow.column_limit = 20;
  EXPECT_EQ(FormatText(helper, "fn F() {\n  //==========================\n}\n",
                       narrow),
            "fn F() {\n  //==========================\n}\n");
}

TEST(FormatterTest, StyleOptions) {
  Testing::CompileHelper helper;
  Style indent4;
  indent4.indent_width = 4;
  EXPECT_EQ(FormatText(helper, "fn F() {\n  return x;\n}\n", indent4),
            "fn F() {\n    return x;\n}\n");

  Style narrow;
  narrow.column_limit = 20;
  EXPECT_LT(
      llvm::count(FormatText(helper, "fn F() {\n  foo(aaaa, bbbb, cccc);\n}\n"),
                  '\n'),
      llvm::count(
          FormatText(helper, "fn F() {\n  foo(aaaa, bbbb, cccc);\n}\n", narrow),
          '\n'));

  Style keep_two;
  keep_two.max_empty_lines_to_keep = 2;
  llvm::StringRef blanks =
      "fn F() {\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n}\n";
  EXPECT_EQ(FormatText(helper, blanks, keep_two), blanks);

  Style no_align;
  no_align.align_trailing_comments = false;
  llvm::StringRef comments = "var x: i32 = 1; // a\nvar yyyy: i32 = 22; // b\n";
  EXPECT_EQ(FormatText(helper, comments, no_align), comments);
}

}  // namespace
}  // namespace Carbon::Format

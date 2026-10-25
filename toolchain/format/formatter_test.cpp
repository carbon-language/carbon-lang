// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <string>

#include "common/raw_string_ostream.h"
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
           Case{"f{\"\"\"// \"}", "f{\"\" \"// \"}\n"},
       }) {
    std::string formatted = FormatText(helper, c.input);
    EXPECT_EQ(formatted, c.expected);
    EXPECT_FALSE(helper.GetTokenizedBuffer(formatted).has_errors());
    EXPECT_EQ(FormatText(helper, formatted), formatted);
  }
}

TEST(FormatterTest, StyleOptions) {
  Testing::CompileHelper helper;
  Style indent4;
  indent4.indent_width = 4;
  EXPECT_EQ(FormatText(helper, "fn F() {\n  return x;\n}\n", indent4),
            "fn F() {\n    return x;\n}\n");

  Style keep_two;
  keep_two.max_empty_lines_to_keep = 2;
  llvm::StringRef blanks =
      "fn F() {\n  var x: i32 = 1;\n\n\n  var y: i32 = 2;\n}\n";
  EXPECT_EQ(FormatText(helper, blanks, keep_two), blanks);
}

}  // namespace
}  // namespace Carbon::Format

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/format/format.h"
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

}  // namespace
}  // namespace Carbon::Format

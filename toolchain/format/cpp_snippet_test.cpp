// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/cpp_snippet.h"

#include <gtest/gtest.h>

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "toolchain/format/style.h"

namespace Carbon::Format {
namespace {

TEST(CppSnippetTest, ReformatsBasicSnippet) {
  EXPECT_EQ(CppSnippet("'''cpp\nint x=1+2;\n'''", 0, Style()),
            "'''cpp\nint x = 1 + 2;\n'''");
}

TEST(CppSnippetTest, IndentsBodyAndClosingDelimiter) {
  EXPECT_EQ(CppSnippet("'''cpp\nint x=1+2;\n'''", 2, Style()),
            "'''cpp\n  int x = 1 + 2;\n  '''");
}

TEST(CppSnippetTest, DeIndentsSourceBeforeReformatting) {
  // The source body and closing `'''` are indented four columns; the result
  // re-indents to the requested two.
  EXPECT_EQ(CppSnippet("'''cpp\n    int x=1;\n    '''", 2, Style()),
            "'''cpp\n  int x = 1;\n  '''");
}

TEST(CppSnippetTest, PointersBindLeft) {
  // Pointer alignment is pinned left, matching Carbon's own C++ style; a
  // consistently right-aligned snippet is still rewritten rather than having
  // its alignment derived and kept.
  EXPECT_EQ(CppSnippet("'''cpp\nint *p = &x;\nint *q = &y;\n'''", 0, Style()),
            "'''cpp\nint* p = &x;\nint* q = &y;\n'''");
}

TEST(CppSnippetTest, AcceptsCommonCppIndicators) {
  EXPECT_TRUE(CppSnippet("'''cc\nint x=1;\n'''", 0, Style()).has_value());
  EXPECT_TRUE(CppSnippet("'''hpp\nint x=1;\n'''", 0, Style()).has_value());
  // The indicator match is case-insensitive.
  EXPECT_TRUE(CppSnippet("'''CPP\nint x=1;\n'''", 0, Style()).has_value());
}

TEST(CppSnippetTest, StripsIntroducerTrailingComment) {
  // A trailing comment on the introducer line ends the file type indicator the
  // way trailing whitespace does; the body is still reformatted and the
  // introducer line -- comment included -- is preserved verbatim.
  EXPECT_EQ(CppSnippet("'''cpp // a comment\nint x=1+2;\n'''", 0, Style()),
            "'''cpp // a comment\nint x = 1 + 2;\n'''");
  // A `//` not followed by whitespace does not begin a comment, so here the
  // indicator is the full `cpp //odd` text, which does not name C++.
  EXPECT_EQ(CppSnippet("'''cpp //odd\nint x=1+2;\n'''", 0, Style()),
            std::nullopt);
}

TEST(CppSnippetTest, IgnoresNonCppLiterals) {
  // A non-C++ indicator, an absent indicator, and a single-line literal are all
  // left alone.
  EXPECT_EQ(CppSnippet("'''text\nhello world\n'''", 0, Style()), std::nullopt);
  EXPECT_EQ(CppSnippet("'''\nhello\n'''", 0, Style()), std::nullopt);
  EXPECT_EQ(CppSnippet("\"int x=1;\"", 0, Style()), std::nullopt);
}

TEST(CppSnippetTest, FormatsUntaggedBodyWhenForced) {
  // An untagged block is left alone normally, but `force_cpp` (the `inline Cpp`
  // case) reformats it as C++ regardless of its indicator.
  EXPECT_EQ(CppSnippet("'''\nint x=1+2;\n'''", 0, Style(),
                       /*force_cpp=*/false),
            std::nullopt);
  EXPECT_EQ(CppSnippet("'''\nint x=1+2;\n'''", 0, Style(),
                       /*force_cpp=*/true),
            "'''\nint x = 1 + 2;\n'''");
  // A forced literal's introducer line may hold a trailing comment too, kept
  // verbatim.
  EXPECT_EQ(CppSnippet("''' // a note\nint x=1+2;\n'''", 0, Style(),
                       /*force_cpp=*/true),
            "''' // a note\nint x = 1 + 2;\n'''");
}

TEST(CppSnippetTest, IgnoresBodyWithDelimiter) {
  // A `'''` in the body could re-form a closing delimiter once re-indented, so
  // such a snippet is left alone.
  EXPECT_EQ(CppSnippet("'''cpp\n// has ''' inside\nint x=1;\n'''", 0, Style()),
            std::nullopt);
}

TEST(CppSnippetTest, IgnoresRawLiteral) {
  // A `#`-raw multi-line literal has different escaping rules; it is left
  // alone.
  EXPECT_EQ(CppSnippet("#'''cpp\nint x=1;\n'''#", 0, Style()), std::nullopt);
}

TEST(CppSnippetTest, AppliesColumnLimitMinusIndent) {
  // clang-format formats to `column_limit - indent`, so the indent eating into
  // the available width makes a call that fits at indent 0 wrap at a larger
  // indent (and thus produce more lines).
  Style style;
  style.column_limit = 40;
  constexpr llvm::StringLiteral Snippet =
      "'''cpp\nreturn foooo(aaaa, bbbb, cccc, dddd);\n'''";
  std::optional<std::string> at_0 = CppSnippet(Snippet, 0, style);
  std::optional<std::string> at_10 = CppSnippet(Snippet, 10, style);
  ASSERT_TRUE(at_0.has_value());
  ASSERT_TRUE(at_10.has_value());
  EXPECT_LT(llvm::count(*at_0, '\n'), llvm::count(*at_10, '\n'));
}

TEST(CppSnippetTest, IgnoresEmptyBody) {
  EXPECT_EQ(CppSnippet("'''cpp\n'''", 0, Style()), std::nullopt);
}

TEST(CppSnippetTest, IgnoresBodyWithControlCharacters) {
  // clang-format mishandles control characters, so a body with one (here a
  // `0x01`) is left unformatted.
  EXPECT_EQ(CppSnippet("'''cpp\nint x=\x01;\n'''", 0, Style()), std::nullopt);
}

TEST(CppSnippetTest, IgnoresBodyWithTab) {
  // clang-format preserves a literal tab inside a string or character literal,
  // but a literal tab is invalid in Carbon multi-line string content, so a body
  // containing one is left unformatted rather than re-encoded into an invalid
  // literal.
  EXPECT_EQ(CppSnippet("'''cpp\nint x=1;\t\n'''", 0, Style()), std::nullopt);
}

TEST(CppSnippetTest, IgnoresMisindentedBody) {
  // The body line is indented less than the closing `'''`, which the lexer
  // would reject; the formatter leaves it untouched rather than guess.
  EXPECT_EQ(CppSnippet("'''cpp\nint x=1;\n  '''", 0, Style()), std::nullopt);
}

}  // namespace
}  // namespace Carbon::Format

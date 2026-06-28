// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/comment.h"

#include <gtest/gtest.h>

#include <string>

namespace Carbon::Format {
namespace {

TEST(CommentTextTest, SingleLineReindented) {
  EXPECT_EQ(CommentText("// hello\n", 2, 80), "  // hello");
}

TEST(CommentTextTest, StripsTrailingWhitespace) {
  EXPECT_EQ(CommentText("// hello   \n", 0, 80), "// hello");
}

TEST(CommentTextTest, ReindentsEveryLineOfABlock) {
  // The block's first line has no leading indentation; the rest carry their
  // original indentation. All are re-indented uniformly to the target.
  EXPECT_EQ(CommentText("// one\n      // two\n  // three\n", 2, 80),
            "  // one\n  // two\n  // three");
}

TEST(CommentTextTest, PreservesInternalSpacingWhenItFits) {
  // Spacing after the `//` and within the content is content, not indentation,
  // so a line that fits is kept verbatim.
  EXPECT_EQ(CommentText("//   - a list item\n", 2, 80), "  //   - a list item");
}

TEST(CommentTextTest, EmptyCommentLinePreserved) {
  EXPECT_EQ(CommentText("//\n", 4, 80), "    //");
}

TEST(CommentTextTest, WrapsAnOverlongLineAtWords) {
  // At a column limit of 20 and indent 0, "// aaaa bbbb cccc" (17) fits but
  // adding " dddd" would reach 22, so it wraps.
  EXPECT_EQ(CommentText("// aaaa bbbb cccc dddd\n", 0, 20),
            "// aaaa bbbb cccc\n// dddd");
}

TEST(CommentTextTest, WrapAccountsForIndent) {
  // The same content at indent 8 has less room (12 columns past the `//`), so
  // each wrapped line holds fewer words: "// aaaa bbbb" reaches column 20.
  EXPECT_EQ(CommentText("// aaaa bbbb cccc dddd\n", 8, 20),
            "        // aaaa bbbb\n        // cccc dddd");
}

TEST(CommentTextTest, LongWordIsNotBroken) {
  // A single word too long for the line is left on its own over-long line.
  EXPECT_EQ(CommentText("// short aaaaaaaaaaaaaaaaaaaaaaaaaaaa end\n", 0, 20),
            "// short\n// aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n// end");
}

TEST(CommentTextTest, WrapKeepsThePrefixAndInteriorSpacing) {
  // The whitespace after `//` is the line's prefix, repeated on each wrapped
  // line (so an indented bullet stays a bullet), and interior runs of spaces
  // in the retained text are kept verbatim, matching clang-format.
  EXPECT_EQ(CommentText("//   - aa bb cc dd\n", 0, 12),
            "//   - aa bb\n//   cc dd");
  EXPECT_EQ(CommentText("// aa  bb   cc dd\n", 0, 13), "// aa  bb\n// cc dd");
}

TEST(CommentTextTest, NonWrappableCommentKeptVerbatim) {
  // A `//` not followed by whitespace (a divider or a lexically invalid
  // comment kept best-effort) is re-indented but never word-wrapped, which
  // would corrupt it (`//======` must not become `// ======`).
  EXPECT_EQ(CommentText("//==========================\n", 2, 20),
            "  //==========================");
  EXPECT_EQ(CommentText("///doc-like comment that runs long\n", 0, 20),
            "///doc-like comment that runs long");
}

TEST(CommentTextTest, WrappedOutputIsStable) {
  // Re-formatting the wrapped output is a no-op: every produced line already
  // fits, so the text is a fixed point.
  std::string once = CommentText("// aaaa bbbb cccc dddd\n", 0, 20);
  EXPECT_EQ(CommentText(once, 0, 20), once);
}

}  // namespace
}  // namespace Carbon::Format

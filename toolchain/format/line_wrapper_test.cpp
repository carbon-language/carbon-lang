// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/line_wrapper.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/token_info.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Format {
namespace {

using ::testing::ElementsAre;

// A solver input built by lexing `source`: the line is the source's tokens
// (skipping the file start and end tokens), with kinds read from the lexed
// buffer the way the formatter reads them. Widths default to the token text
// and are overridden through `Info` to model wide tokens; roles and the other
// annotations are set the same way. `drop_prefix` drops leading tokens from
// the line, letting a well-formed source hand the solver a malformed line
// shape (such as a stray `)`).
struct TestLine {
  explicit TestLine(llvm::StringRef source, int drop_prefix = 0)
      : buffer(helper.GetTokenizedBuffer(source)),
        token_infos(
            TokenInfoStore::MakeWithExplicitSize(buffer.size(), TokenInfo())) {
    for (auto token : buffer.tokens()) {
      token_infos.Get(token).column_width =
          static_cast<int>(buffer.GetTokenText(token).size());
      if (!buffer.GetKind(token).IsOneOf(
              {Lex::TokenKind::FileStart, Lex::TokenKind::FileEnd})) {
        tokens.push_back(token);
      }
    }
    tokens.erase(tokens.begin(), tokens.begin() + drop_prefix);
  }

  // The formatter's annotations for the line's i-th token, mutable so a test
  // can model wide tokens, roles, and penalties directly.
  auto Info(int i) -> TokenInfo& { return token_infos.Get(tokens[i]); }

  auto Solve(int indent) -> llvm::SmallVector<int> {
    return SolveLineBreaks(buffer, token_infos, tokens, indent);
  }

  Testing::CompileHelper helper;
  Lex::TokenizedBuffer& buffer;
  TokenInfoStore token_infos;
  llvm::SmallVector<Lex::TokenIndex> tokens;
};

TEST(SolveLineBreaksTest, EmptyAndSingleToken) {
  TestLine empty("");
  EXPECT_TRUE(empty.Solve(0).empty());

  TestLine one("aaaaa");
  EXPECT_THAT(one.Solve(0), ElementsAre(-1));
}

TEST(SolveLineBreaksTest, FittingLineHasNoBreaks) {
  // Two short words at indent 0 fit well within the limit, so neither breaks.
  TestLine line("aaa bbb");
  EXPECT_THAT(line.Solve(0), ElementsAre(-1, -1));
}

TEST(SolveLineBreaksTest, BreaksAnOverlongLine) {
  // Two wide words can't share a line (50 + 1 + 50 = 101 > 80), so the second
  // wraps to the statement-level continuation indent (0 + 4).
  TestLine line("a b");
  line.Info(0).column_width = 50;
  line.Info(1).column_width = 50;
  EXPECT_THAT(line.Solve(0), ElementsAre(-1, ContinuationIndentWidth));
}

TEST(SolveLineBreaksTest, ContinuationIndentTracksStatementIndent) {
  // The same overlong pair at indent 2 wraps to 2 + 4 = 6.
  TestLine line("a b");
  line.Info(0).column_width = 50;
  line.Info(1).column_width = 50;
  EXPECT_THAT(line.Solve(2), ElementsAre(-1, 2 + ContinuationIndentWidth));
}

TEST(SolveLineBreaksTest, WrapsArgumentAlignedAfterOpenBracket) {
  // `name(arg1, arg2)` where the one-line form overflows: the wrapped argument
  // aligns just after the `(` (`AlignAfterOpenBracket = Align`), not at the
  // statement continuation indent.
  //
  // `name` ends at column 4; `(` (tight, postfix) ends at 5, so the
  // open-bracket anchor is column 5. The first argument stays on the opening
  // line; the second wraps to column 5.
  TestLine line("name(arg1, arg2)");
  line.Info(1).role = TokenRole::PostfixBracket;
  line.Info(2).column_width = 40;
  line.Info(4).column_width = 40;
  EXPECT_THAT(line.Solve(0), ElementsAre(-1, -1, -1, -1, /*arg2=*/5, -1));
}

TEST(SolveLineBreaksTest, NeverBreaksBeforeAClosingBracket) {
  // Even when the line overflows, the closing `)` hugs the last argument rather
  // than taking its own line (`CanBreakBefore` forbids a break there).
  TestLine line("name(arg)");
  line.Info(1).role = TokenRole::PostfixBracket;
  line.Info(2).column_width = 90;
  llvm::SmallVector<int> breaks = line.Solve(0);
  EXPECT_EQ(breaks.back(), -1) << "closing bracket should not break";
}

TEST(SolveLineBreaksTest, ClosedBracketRevertsToEnclosingAnchor) {
  // `name(arg) other...` where a break lands after the bracket has closed: the
  // wrapped token indents to the statement-level continuation indent, not the
  // (already popped) bracket anchor.
  TestLine line("name(arg) c d");
  line.Info(1).role = TokenRole::PostfixBracket;
  line.Info(4).column_width = 40;
  line.Info(5).column_width = 40;
  EXPECT_THAT(line.Solve(0), ElementsAre(-1, -1, -1, -1, -1,
                                         /*reverted=*/ContinuationIndentWidth));
}

TEST(SolveLineBreaksTest, StrayClosingBracketIsSafe) {
  // Malformed input can present a closing bracket with no opener; the bottom
  // (statement-level) anchor is never popped, so a break still lands there.
  // The line drops the `(` from a well-formed source, leaving a stray `)`.
  TestLine line("() a b", /*drop_prefix=*/1);
  line.Info(1).column_width = 50;
  line.Info(2).column_width = 50;
  EXPECT_THAT(line.Solve(0), ElementsAre(-1, -1, ContinuationIndentWidth));
}

TEST(SolveLineBreaksTest, PrefersTheCheaperBreak) {
  // `lhs = rhs...` that overflows: breaking after `=` (penalty 2) is cheaper
  // than breaking between the two words of the right-hand side (penalty 3), and
  // the right-hand side then fits on one continuation line, so the `=` break is
  // chosen and nothing else wraps.
  TestLine line("lhs = rhs1 rhs2");
  line.Info(0).column_width = 30;
  line.Info(2).column_width = 25;
  line.Info(3).column_width = 25;
  EXPECT_THAT(line.Solve(0),
              ElementsAre(-1, -1, /*after `=`*/ ContinuationIndentWidth, -1));
}

}  // namespace
}  // namespace Carbon::Format

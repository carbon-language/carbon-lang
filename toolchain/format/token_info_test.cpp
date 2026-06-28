// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/token_info.h"

#include <gtest/gtest.h>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/format/style.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Format {
namespace {

// Wraps a lexed source string for layout-decision tests: `Token(i)` is the
// i-th token of `source` (skipping the file start and end tokens), `SetRole`
// annotates a token in the info store the way the formatter would from the
// parse tree, and the remaining helpers run the layout decisions over the
// store. Token widths are filled from the token text, as in the formatter.
class TokenTester {
 public:
  explicit TokenTester(llvm::StringRef source)
      : tokens_(helper_.GetTokenizedBuffer(source)),
        token_infos_(
            TokenInfoStore::MakeWithExplicitSize(tokens_.size(), TokenInfo())) {
    for (auto token : tokens_.tokens()) {
      token_infos_.Get(token).column_width =
          static_cast<int>(tokens_.GetTokenText(token).size());
      if (!tokens_.GetKind(token).IsOneOf(
              {Lex::TokenKind::FileStart, Lex::TokenKind::FileEnd})) {
        source_tokens_.push_back(token);
      }
    }
  }

  auto Token(int i) -> Lex::TokenIndex { return source_tokens_[i]; }

  // The formatter's annotations for the i-th token, mutable so a test can set
  // roles and penalties directly.
  auto Info(int i) -> TokenInfo& { return token_infos_.Get(Token(i)); }

  auto SetRole(int i, TokenRole role) -> void { Info(i).role = role; }

  auto Spaces(int left, int right) -> int {
    return SpacesBefore(tokens_, token_infos_, Token(left), Token(right));
  }

  auto CanBreak(int left, int right) -> bool {
    return CanBreakBefore(tokens_, token_infos_, Token(left), Token(right));
  }

  auto Penalty(int left, int right) -> int {
    return SplitPenalty(tokens_, token_infos_, Token(left), Token(right),
                        style_);
  }

  auto Width() -> int {
    return RenderedWidth(tokens_, token_infos_, source_tokens_);
  }

 private:
  Testing::CompileHelper helper_;
  Lex::TokenizedBuffer& tokens_;
  TokenInfoStore token_infos_;
  llvm::SmallVector<Lex::TokenIndex> source_tokens_;
  // The canonical style, used by the penalty tests below.
  Style style_;
};

TEST(RoleForNodeKindTest, PostfixBrackets) {
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::ExplicitParamListStart),
            TokenRole::PostfixBracket);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::ImplicitParamListStart),
            TokenRole::PostfixBracket);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::CallExprStart),
            TokenRole::PostfixBracket);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::IndexExprStart),
            TokenRole::PostfixBracket);
}

TEST(RoleForNodeKindTest, NotPostfixBrackets) {
  // A control-flow condition `(` and a block `{` are not postfix brackets.
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::IfConditionStart),
            TokenRole::Unknown);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::FunctionDefinitionStart),
            TokenRole::Unknown);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::CodeBlockStart),
            TokenRole::Unknown);
}

TEST(SpacesBeforeTest, PostfixBracketIsTight) {
  // `F(` and `x[` get no space; a grouping `(` after a keyword does.
  TokenTester call("F(x)");
  call.SetRole(1, TokenRole::PostfixBracket);
  EXPECT_EQ(call.Spaces(0, 1), 0);

  TokenTester subscript("x[i]");
  subscript.SetRole(1, TokenRole::PostfixBracket);
  EXPECT_EQ(subscript.Spaces(0, 1), 0);

  TokenTester group("if (c)");
  EXPECT_EQ(group.Spaces(0, 1), 1);
}

TEST(SpacesBeforeTest, SeparatorsAndClosers) {
  TokenTester t("a, b; (c) [d]");
  EXPECT_EQ(t.Spaces(0, 1), 0);  // `a` `,`
  EXPECT_EQ(t.Spaces(2, 3), 0);  // `b` `;`
  EXPECT_EQ(t.Spaces(5, 6), 0);  // `c` `)`
  EXPECT_EQ(t.Spaces(8, 9), 0);  // `d` `]`
}

TEST(SpacesBeforeTest, BindingColon) {
  // `n: T`: no space before the colon, one space after.
  TokenTester t("n: T");
  EXPECT_EQ(t.Spaces(0, 1), 0);
  EXPECT_EQ(t.Spaces(1, 2), 1);
}

TEST(SpacesBeforeTest, NoPaddingInsideBrackets) {
  TokenTester t("(a) [b]");
  EXPECT_EQ(t.Spaces(0, 1), 0);  // `(` `a`
  EXPECT_EQ(t.Spaces(3, 4), 0);  // `[` `b`
}

TEST(SpacesBeforeTest, EmptyBracesAreCompact) {
  TokenTester t("{}");
  EXPECT_EQ(t.Spaces(0, 1), 0);
}

TEST(RoleForNodeKindTest, MemberAccessAndDesignators) {
  // A member-access `.`/`->` binds tightly on both sides; a leading designator
  // `.` acts as a symbolic prefix to the name it introduces. The return-arrow
  // `->` shares the member `->`'s token kind but keeps the default role.
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::MemberAccessExpr),
            TokenRole::MemberAccess);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::PointerMemberAccessExpr),
            TokenRole::MemberAccess);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::DesignatorExpr),
            TokenRole::PrefixOperator);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::StructFieldDesignator),
            TokenRole::PrefixOperator);
  EXPECT_EQ(RoleForNodeKind(Parse::NodeKind::ReturnType), TokenRole::Unknown);
}

TEST(SpacesBeforeTest, MemberAccessIsTight) {
  TokenTester dot("a.b");
  dot.SetRole(1, TokenRole::MemberAccess);
  EXPECT_EQ(dot.Spaces(0, 1), 0);
  EXPECT_EQ(dot.Spaces(1, 2), 0);

  TokenTester arrow("p->x");
  arrow.SetRole(1, TokenRole::MemberAccess);
  EXPECT_EQ(arrow.Spaces(0, 1), 0);
  EXPECT_EQ(arrow.Spaces(1, 2), 0);
}

TEST(SpacesBeforeTest, MemberAccessOnLiteralKeepsItsSpace) {
  // Gluing a member `.` onto a numeric literal would lex the `.` as part of
  // the literal, so the space stays: `2 .rt`, not `2.rt`.
  TokenTester int_literal("2 .rt");
  int_literal.SetRole(1, TokenRole::MemberAccess);
  EXPECT_EQ(int_literal.Spaces(0, 1), 1);
  EXPECT_EQ(int_literal.Spaces(1, 2), 0);

  TokenTester real_literal("2.5 .rt");
  real_literal.SetRole(1, TokenRole::MemberAccess);
  EXPECT_EQ(real_literal.Spaces(0, 1), 1);
}

TEST(SpacesBeforeTest, DesignatorSpacesLikeAPrefix) {
  // `case .Red`: a space before the designator `.`, none after it. After a
  // comma, the comma's following space is kept: `{.a = 1, .b = 2}`.
  TokenTester t("case .Red, .Blue");
  t.SetRole(1, TokenRole::PrefixOperator);
  t.SetRole(4, TokenRole::PrefixOperator);
  EXPECT_EQ(t.Spaces(0, 1), 1);
  EXPECT_EQ(t.Spaces(1, 2), 0);
  EXPECT_EQ(t.Spaces(3, 4), 1);
}

TEST(SpacesBeforeTest, ReturnArrowIsSpaced) {
  // A return-type `->` has no member-access role, so it keeps a space on each
  // side: `) -> i32`.
  TokenTester t("() -> i32");
  EXPECT_EQ(t.Spaces(1, 2), 1);
  EXPECT_EQ(t.Spaces(2, 3), 1);
}

TEST(SpacesBeforeTest, DefaultIsOneSpace) {
  // Two adjacent words, and a binary operator, are space-separated.
  TokenTester t("a b + c");
  EXPECT_EQ(t.Spaces(0, 1), 1);
  EXPECT_EQ(t.Spaces(1, 2), 1);
}

TEST(CanBreakBeforeTest, NeverBeforeSeparatorsOrClosers) {
  TokenTester t("F(a, b); c[d]; {}");
  EXPECT_FALSE(t.CanBreak(2, 3));    // Before `,`.
  EXPECT_FALSE(t.CanBreak(4, 5));    // Before `)`.
  EXPECT_FALSE(t.CanBreak(5, 6));    // Before `;`.
  EXPECT_FALSE(t.CanBreak(9, 10));   // Before `]`.
  EXPECT_FALSE(t.CanBreak(12, 13));  // Before `}`.
}

TEST(CanBreakBeforeTest, NeverBeforePostfixBracket) {
  TokenTester t("F(x)");
  t.SetRole(1, TokenRole::PostfixBracket);
  EXPECT_FALSE(t.CanBreak(0, 1));
}

TEST(CanBreakBeforeTest, AllowedElsewhere) {
  TokenTester t("a b, c");
  EXPECT_TRUE(t.CanBreak(0, 1));  // Between two words.
  EXPECT_TRUE(t.CanBreak(2, 3));  // After a comma.
}

TEST(CanBreakBeforeTest, NeverBeforeEqualOrInfixOperator) {
  TokenTester t("a = b + c");
  // A `=` keeps to the previous line; the break goes after it.
  EXPECT_FALSE(t.CanBreak(0, 1));
  // An infix operator (marked by `break_penalty_after`) is the same: break
  // after it, before the right operand.
  t.Info(3).break_penalty_after = 13;
  EXPECT_FALSE(t.CanBreak(2, 3));
}

TEST(CanBreakBeforeTest, NeverBeforeBindingColon) {
  // The colon hugs its binding name; a break may go after the colon (at a
  // steep penalty), never before it.
  TokenTester t("name: Type");
  EXPECT_FALSE(t.CanBreak(0, 1));
  EXPECT_TRUE(t.CanBreak(1, 2));
}

TEST(CanBreakBeforeTest, NeverSplittingSymbolicUnaryOperators) {
  // Whitespace between a symbolic unary operator and its operand is invalid
  // Carbon (the lexer's fixity rules reject it), so a break after a symbolic
  // prefix operator or before a postfix operator would corrupt the code.
  TokenTester t("F(&a) p*");
  t.SetRole(2, TokenRole::PrefixOperator);
  EXPECT_FALSE(t.CanBreak(2, 3));
  t.SetRole(6, TokenRole::PostfixOperator);
  EXPECT_FALSE(t.CanBreak(5, 6));
}

TEST(SplitPenaltyTest, WordPrefixOperator) {
  // A word prefix operator keeps its operand-side space, so a break there is
  // legal but priced like clang-format's unary-operator penalty.
  TokenTester t("not flag");
  t.SetRole(0, TokenRole::PrefixOperator);
  EXPECT_TRUE(t.CanBreak(0, 1));
  EXPECT_EQ(t.Penalty(0, 1), 60);
}

TEST(SplitPenaltyTest, BindingColonAndDeclKeywords) {
  TokenTester t("private fn F(name: Type) -> R");
  // After a binding colon: a near-last resort, above the return-type break.
  EXPECT_EQ(t.Penalty(5, 6), 100);
  // After a keyword: keywords bind to the construct they introduce or modify,
  // so splitting one from what follows is the costliest legal break.
  EXPECT_EQ(t.Penalty(0, 1), 200);
  EXPECT_EQ(t.Penalty(1, 2), 200);
}

TEST(OperatorInfoTest, BinaryOperators) {
  // Tighter-binding operators get a larger break penalty, so the loosest breaks
  // first. Non-assignment operators align their operands; assignments do not.
  Style style;
  auto mul = OperatorInfoForNodeKind(Parse::NodeKind::InfixOperatorStar, style);
  EXPECT_EQ(mul.break_penalty, 14);
  EXPECT_TRUE(mul.aligns_operands);

  auto add = OperatorInfoForNodeKind(Parse::NodeKind::InfixOperatorPlus, style);
  EXPECT_EQ(add.break_penalty, 13);
  EXPECT_LT(add.break_penalty, mul.break_penalty);

  auto logical_and =
      OperatorInfoForNodeKind(Parse::NodeKind::ShortCircuitOperatorAnd, style);
  EXPECT_TRUE(logical_and.aligns_operands);
  EXPECT_LT(logical_and.break_penalty, add.break_penalty);

  auto assign =
      OperatorInfoForNodeKind(Parse::NodeKind::InfixOperatorEqual, style);
  EXPECT_EQ(assign.break_penalty, style.penalty_break_assignment);
  EXPECT_FALSE(assign.aligns_operands);

  // A non-operator node is not classified.
  EXPECT_EQ(
      OperatorInfoForNodeKind(Parse::NodeKind::IntLiteral, style).break_penalty,
      -1);
}

TEST(SplitPenaltyTest, InfixOperatorUsesItsBreakPenalty) {
  TokenTester t("a + b");
  t.Info(1).break_penalty_after = 13;
  EXPECT_EQ(t.Penalty(1, 2), 13);
}

TEST(SplitPenaltyTest, Values) {
  Style style;
  TokenTester t("a, b = c (d) e.f g h");
  // Cheap: after a comma or `=`, or the first element after a bracket.
  EXPECT_EQ(t.Penalty(1, 2), 1);
  EXPECT_EQ(t.Penalty(3, 4), style.penalty_break_assignment);
  EXPECT_EQ(t.Penalty(5, 6), style.penalty_break_before_first_call_parameter);
  // Expensive: before a member access.
  EXPECT_EQ(t.Penalty(8, 9), 150);
  // Default.
  EXPECT_EQ(t.Penalty(11, 12), 3);
}

TEST(RenderedWidthTest, SumsWidthsAndSpaces) {
  // `fn F()` is 6 columns: widths 2 + 1 + 1 + 1, plus a single space after `fn`
  // (none before the call paren or before the close paren).
  TokenTester t("fn F()");
  t.SetRole(2, TokenRole::PostfixBracket);
  EXPECT_EQ(t.Width(), 6);
}

}  // namespace
}  // namespace Carbon::Format

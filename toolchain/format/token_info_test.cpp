// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/token_info.h"

#include <gtest/gtest.h>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/testing/compile_helper.h"

namespace Carbon::Format {
namespace {

// Wraps a lexed source string for spacing tests: `Token(i)` is the i-th token
// of `source` (skipping the file start and end tokens), `SetRole` annotates a
// token in the info store the way the formatter would from the parse tree, and
// `Spaces` runs `SpacesBefore` over the store.
class TokenTester {
 public:
  explicit TokenTester(llvm::StringRef source)
      : tokens_(helper_.GetTokenizedBuffer(source)),
        token_infos_(
            TokenInfoStore::MakeWithExplicitSize(tokens_.size(), TokenInfo())) {
    for (auto token : tokens_.tokens()) {
      if (!tokens_.GetKind(token).IsOneOf(
              {Lex::TokenKind::FileStart, Lex::TokenKind::FileEnd})) {
        source_tokens_.push_back(token);
      }
    }
  }

  auto Token(int i) -> Lex::TokenIndex { return source_tokens_[i]; }

  auto SetRole(int i, TokenRole role) -> void {
    token_infos_.Get(Token(i)).role = role;
  }

  auto Spaces(int left, int right) -> int {
    return SpacesBefore(tokens_, token_infos_, Token(left), Token(right));
  }

 private:
  Testing::CompileHelper helper_;
  Lex::TokenizedBuffer& tokens_;
  TokenInfoStore token_infos_;
  llvm::SmallVector<Lex::TokenIndex> source_tokens_;
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

}  // namespace
}  // namespace Carbon::Format

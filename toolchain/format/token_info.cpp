// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/token_info.h"

#include <optional>

namespace Carbon::Format {

auto RoleForNodeKind(Parse::NodeKind kind) -> TokenRole {
  switch (kind) {
    case Parse::NodeKind::ExplicitParamListStart:
    case Parse::NodeKind::ImplicitParamListStart:
    case Parse::NodeKind::CallExprStart:
    case Parse::NodeKind::IndexExprStart:
      return TokenRole::PostfixBracket;

    case Parse::NodeKind::MemberAccessExpr:
    case Parse::NodeKind::PointerMemberAccessExpr:
      return TokenRole::MemberAccess;

      // A leading designator `.` (`.Red`, `{.x = 1}`, `.Self`) introduces the
      // name after it like a symbolic prefix operator: tight to the name,
      // ordinary spacing before.
    case Parse::NodeKind::DesignatorExpr:
    case Parse::NodeKind::StructFieldDesignator:
      return TokenRole::PrefixOperator;

      // Prefix unary operators, for example `*p`, `-x`, `&v`, and `not b`.
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name) \
  case Parse::NodeKind::PrefixOperator##Name:
#include "toolchain/parse/node_kind.def"
      return TokenRole::PrefixOperator;

      // Postfix unary operators, for example the pointer type `p*`.
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name) \
  case Parse::NodeKind::PostfixOperator##Name:
#include "toolchain/parse/node_kind.def"
      return TokenRole::PostfixOperator;

    default:
      return TokenRole::Unknown;
  }
}

auto SpacesBefore(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right) -> int {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  Lex::TokenKind right_kind = tokens.GetKind(right);
  TokenRole left_role = token_infos.Get(left).role;
  TokenRole right_role = token_infos.Get(right).role;
  // A call, parameter-list, or subscript bracket binds tightly to the token it
  // follows, for example `F(` or `x[`.
  if (right_role == TokenRole::PostfixBracket) {
    return 0;
  }
  // A postfix unary operator binds tightly to its operand: `p*`, not `p *`.
  if (right_role == TokenRole::PostfixOperator) {
    return 0;
  }
  // A symbolic prefix unary operator binds tightly to its operand: `*p`, `-x`,
  // `&v`. Word-based prefix operators such as `not` and `const` keep a space,
  // which the default below provides.
  if (left_role == TokenRole::PrefixOperator && left_kind.is_symbol()) {
    return 0;
  }
  // No space before a separator or a closing bracket.
  if (right_kind.IsOneOf({Lex::TokenKind::Comma, Lex::TokenKind::Semi,
                          Lex::TokenKind::CloseParen,
                          Lex::TokenKind::CloseSquareBracket})) {
    return 0;
  }
  // No space before a binding colon: `name: Type`.
  if (right_kind == Lex::TokenKind::Colon) {
    return 0;
  }
  // No padding just inside `(` or `[`.
  if (left_kind.IsOneOf(
          {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket})) {
    return 0;
  }
  // An empty `{}` stays compact.
  if (left_kind == Lex::TokenKind::OpenCurlyBrace &&
      right_kind == Lex::TokenKind::CloseCurlyBrace) {
    return 0;
  }
  // Member access binds tightly on both sides: `a.b`, `p->x`. A `.` or `->`
  // without this role (a designator `.` or a return-type `->`) follows the
  // other rules instead. One lexing hazard: after a numeric literal the `.`
  // must keep its space (`2 .rt`), since glued it would lex as part of the
  // literal and change the token sequence.
  if (right_role == TokenRole::MemberAccess &&
      right_kind == Lex::TokenKind::Period &&
      left_kind.IsOneOf(
          {Lex::TokenKind::IntLiteral, Lex::TokenKind::RealLiteral})) {
    return 1;
  }
  if (left_role == TokenRole::MemberAccess ||
      right_role == TokenRole::MemberAccess) {
    return 0;
  }
  return 1;
}

auto CanBreakBefore(const Lex::TokenizedBuffer& tokens,
                    const TokenInfoStore& token_infos, Lex::TokenIndex /*left*/,
                    Lex::TokenIndex right) -> bool {
  // Separators and closing brackets hug the preceding token, so a break before
  // them is never allowed.
  if (tokens.GetKind(right).IsOneOf(
          {Lex::TokenKind::Comma, Lex::TokenKind::Semi,
           Lex::TokenKind::CloseParen, Lex::TokenKind::CloseSquareBracket,
           Lex::TokenKind::CloseCurlyBrace})) {
    return false;
  }
  // A call, parameter-list, or subscript bracket stays with its callee or name.
  if (token_infos.Get(right).role == TokenRole::PostfixBracket) {
    return false;
  }
  // Anywhere else a break is allowed; the split penalty steers where breaks
  // actually land.
  return true;
}

auto SplitPenalty(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& /*token_infos*/, Lex::TokenIndex left,
                  Lex::TokenIndex right) -> int {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  // Cheap, encouraged breaks.
  if (left_kind == Lex::TokenKind::Comma) {
    // After a comma, between list elements.
    return 1;
  }
  if (left_kind == Lex::TokenKind::Equal) {
    // After `=`, onto the right-hand side of an assignment.
    return 2;
  }
  if (left_kind.IsOneOf(
          {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket})) {
    // After an opening bracket, before the first element.
    return 19;
  }
  // Expensive, discouraged breaks.
  if (tokens.GetKind(right) == Lex::TokenKind::Period) {
    // Before a member access in a chain.
    return 150;
  }
  // Default.
  return 3;
}

auto RenderedWidth(const Lex::TokenizedBuffer& tokens,
                   const TokenInfoStore& token_infos,
                   llvm::ArrayRef<Lex::TokenIndex> line) -> int {
  int width = 0;
  std::optional<Lex::TokenIndex> previous;
  for (Lex::TokenIndex token : line) {
    if (previous) {
      width += SpacesBefore(tokens, token_infos, *previous, token);
    }
    width += token_infos.Get(token).column_width;
    previous = token;
  }
  return width;
}

}  // namespace Carbon::Format

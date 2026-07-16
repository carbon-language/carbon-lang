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

auto OperatorInfoForNodeKind(Parse::NodeKind kind, const Style& style)
    -> OperatorInfo {
  // The break penalty is the operator's precedence as a "looseness" rank: a
  // lower value for a looser-binding operator, so the solver breaks the loosest
  // operator in an expression first. The values mirror Carbon's precedence
  // ladder (see toolchain/parse/precedence.h) the way clang-format uses the
  // precedence level as the split penalty. They are starting values, tuned
  // against the test corpus rather than fixed by policy.
  switch (kind) {
    // Assignment, including compound assignment: the right-hand side is
    // continuation-indented, not operand-aligned.
    case Parse::NodeKind::InfixOperatorEqual:
    case Parse::NodeKind::InfixOperatorPlusEqual:
    case Parse::NodeKind::InfixOperatorMinusEqual:
    case Parse::NodeKind::InfixOperatorStarEqual:
    case Parse::NodeKind::InfixOperatorSlashEqual:
    case Parse::NodeKind::InfixOperatorPercentEqual:
    case Parse::NodeKind::InfixOperatorAmpEqual:
    case Parse::NodeKind::InfixOperatorPipeEqual:
    case Parse::NodeKind::InfixOperatorCaretEqual:
    case Parse::NodeKind::InfixOperatorLessLessEqual:
    case Parse::NodeKind::InfixOperatorGreaterGreaterEqual:
      return {.break_penalty = style.penalty_break_assignment,
              .aligns_operands = false};
    // Logical, loosest of the operand-aligning operators.
    case Parse::NodeKind::ShortCircuitOperatorOr:
      return {.break_penalty = 4, .aligns_operands = true};
    case Parse::NodeKind::ShortCircuitOperatorAnd:
      return {.break_penalty = 5, .aligns_operands = true};
    // Bitwise.
    case Parse::NodeKind::InfixOperatorPipe:
      return {.break_penalty = 6, .aligns_operands = true};
    case Parse::NodeKind::InfixOperatorCaret:
      return {.break_penalty = 7, .aligns_operands = true};
    case Parse::NodeKind::InfixOperatorAmp:
      return {.break_penalty = 8, .aligns_operands = true};
    // Equality and relational.
    case Parse::NodeKind::InfixOperatorEqualEqual:
    case Parse::NodeKind::InfixOperatorExclaimEqual:
      return {.break_penalty = 9, .aligns_operands = true};
    case Parse::NodeKind::InfixOperatorLess:
    case Parse::NodeKind::InfixOperatorLessEqual:
    case Parse::NodeKind::InfixOperatorGreater:
    case Parse::NodeKind::InfixOperatorGreaterEqual:
    case Parse::NodeKind::InfixOperatorLessEqualGreater:
      return {.break_penalty = 10, .aligns_operands = true};
    // Cast.
    case Parse::NodeKind::InfixOperatorAs:
      return {.break_penalty = 11, .aligns_operands = true};
    // Bit shift.
    case Parse::NodeKind::InfixOperatorLessLess:
    case Parse::NodeKind::InfixOperatorGreaterGreater:
      return {.break_penalty = 12, .aligns_operands = true};
    // Additive.
    case Parse::NodeKind::InfixOperatorPlus:
    case Parse::NodeKind::InfixOperatorMinus:
      return {.break_penalty = 13, .aligns_operands = true};
    // Multiplicative, tightest binding.
    case Parse::NodeKind::InfixOperatorStar:
    case Parse::NodeKind::InfixOperatorSlash:
    case Parse::NodeKind::InfixOperatorPercent:
      return {.break_penalty = 14, .aligns_operands = true};
    default:
      return {.break_penalty = -1, .aligns_operands = false};
  }
}

auto MemberAccessBreakPenalty(Parse::NodeKind kind) -> int {
  switch (kind) {
    case Parse::NodeKind::MemberAccessExpr:
    case Parse::NodeKind::PointerMemberAccessExpr:
      // Breaking a member-access chain is expensive (clang-format's
      // `isMemberAccess` penalty), so the solver wraps only when it must, and
      // prefers breaking elsewhere (after `=`, between arguments) first. This
      // is the penalty for a non-final link; the formatter lowers the last link
      // in each chain to the cheaper 35 (see `Formatter`'s constructor).
      return 150;
    default:
      return -1;
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
                    const TokenInfoStore& token_infos, Lex::TokenIndex left,
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
  // Never break before `=`: assignments and initializers keep the `=` on the
  // left and break after it.
  if (tokens.GetKind(right) == Lex::TokenKind::Equal) {
    return false;
  }
  // Never break before a binding colon: the colon hugs its name, and a break
  // goes after it instead.
  if (tokens.GetKind(right) == Lex::TokenKind::Colon) {
    return false;
  }
  // Never separate a symbolic unary operator from its operand: the lexer's
  // fixity rules forbid whitespace on the operand side, so a break after a
  // symbolic prefix operator or before a postfix operator would turn valid
  // code into invalid code.
  if (token_infos.Get(left).role == TokenRole::PrefixOperator &&
      tokens.GetKind(left).is_symbol()) {
    return false;
  }
  if (token_infos.Get(right).role == TokenRole::PostfixOperator) {
    return false;
  }
  // Never break before an infix operator. With break-after-operator style the
  // split point is after the operator, before its right operand.
  if (token_infos.Get(right).break_penalty_after >= 0) {
    return false;
  }
  // Never break after a member-access `.`/`->`: the member stays attached, so a
  // chain wraps before the `.`/`->`, not after it.
  if (tokens.GetKind(left).IsOneOf(
          {Lex::TokenKind::Period, Lex::TokenKind::MinusGreater})) {
    return false;
  }
  // Anywhere else a break is allowed; the split penalty steers where breaks
  // actually land.
  return true;
}

auto SplitPenalty(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right, const Style& style) -> int {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  // Cheap, encouraged breaks.
  if (left_kind == Lex::TokenKind::Comma) {
    // After a comma, between list elements.
    return 1;
  }
  // After an infix operator, before its right operand. The penalty is the
  // operator's precedence-based break penalty, so the loosest operator in an
  // expression is broken first.
  int left_break_penalty_after = token_infos.Get(left).break_penalty_after;
  if (left_break_penalty_after >= 0) {
    return left_break_penalty_after;
  }
  // Before a member-access `.`/`->` that the parse tree identified.
  int right_break_penalty_before = token_infos.Get(right).break_penalty_before;
  if (right_break_penalty_before >= 0) {
    return right_break_penalty_before;
  }
  if (left_kind == Lex::TokenKind::Equal) {
    // After a `=` initializer that is not an infix-operator node (so has no
    // `break_penalty_after`), onto the right-hand side.
    return style.penalty_break_assignment;
  }
  if (left_kind.IsOneOf(
          {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket})) {
    // After an opening bracket, before the first element.
    return style.penalty_break_before_first_call_parameter;
  }
  // Expensive, discouraged breaks.
  if (token_infos.Get(left).role == TokenRole::PrefixOperator) {
    // After a word prefix operator such as `not`, before its operand
    // (clang-format's unary-operator penalty). A symbolic prefix operator
    // never reaches here: `CanBreakBefore` forbids that break outright.
    return 60;
  }
  if (left_kind == Lex::TokenKind::Colon) {
    // After a binding colon, moving the type off the line that names it. A
    // near-last resort, but cheaper than splitting at a keyword, so a
    // declaration nothing else can save wraps here.
    return 100;
  }
  if (tokens.GetKind(right) == Lex::TokenKind::Period) {
    // Before a `.` not otherwise classified (for example a struct-literal
    // field), matching the member-access penalty.
    return 150;
  }
  if (tokens.GetKind(right) == Lex::TokenKind::MinusGreater) {
    // Before a `->` not classified as a pointer member access: the function
    // return type onto its own line (`PenaltyReturnTypeOnItsOwnLine`).
    return 60;
  }
  if (left_kind.is_keyword()) {
    // After any other keyword. A keyword binds to the construct it introduces
    // or modifies (`private` and `fn` to their declaration, `return` to its
    // operand, `while` to its condition), and a break there moves the
    // construct to the continuation indent, left of where it started, which
    // reads as a new statement. This is the costliest legal break, the analog
    // of clang-format's penalty against separating a declaration's specifiers
    // from its name (`TT_StartOfName`). Word operators are already classified
    // above: infix ones break by precedence, and prefix ones cost the
    // unary-operator penalty.
    return 200;
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

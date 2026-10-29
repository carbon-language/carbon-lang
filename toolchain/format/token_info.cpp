// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/token_info.h"

#include <cstddef>
#include <optional>
#include <string>

#include "common/check.h"
#include "toolchain/parse/precedence.h"

namespace Carbon::Format {

// Returns true if placing `left` and `right` adjacent with no space would cause
// re-lexing to merge or re-split them into different tokens.
static auto WouldMergeTokens(const Lex::TokenizedBuffer& tokens,
                             const TokenInfoStore& token_infos,
                             Lex::TokenIndex left, Lex::TokenIndex right)
    -> bool {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  Lex::TokenKind right_kind = tokens.GetKind(right);

  // An integer literal followed immediately by `.` and an alphanumeric
  // character lexes as a real literal (`2 .rt` vs `2.rt`), unless the integer
  // literal itself immediately follows a `.` or `->` member access (`t.0.1`,
  // `p->0.1`), where `Lexer::CanFormRealLiteral` disables real-literal lexing.
  if (left_kind == Lex::TokenKind::IntLiteral &&
      right_kind == Lex::TokenKind::Period) {
    if (left.index > 0) {
      Lex::TokenIndex prev(left.index - 1);
      if (token_infos.Get(prev).role == TokenRole::MemberAccess &&
          tokens.GetKind(prev).IsOneOf(
              {Lex::TokenKind::Period, Lex::TokenKind::MinusGreater}) &&
          !tokens.IsRecoveryToken(prev) && !tokens.IsRecoveryToken(left)) {
        int prev_end =
            tokens.GetByteOffset(prev) + tokens.GetTokenText(prev).size();
        int left_start = tokens.GetByteOffset(left);
        if (left_start >= prev_end && !tokens.source()
                                           .text()
                                           .slice(prev_end, left_start)
                                           .contains("//")) {
          return false;
        }
      }
    }
    return true;
  }

  if (!left_kind.is_symbol() || !right_kind.is_symbol() ||
      left_kind.is_grouping_symbol() || right_kind.is_grouping_symbol() ||
      left_kind.IsOneOf({Lex::TokenKind::Comma, Lex::TokenKind::Semi}) ||
      right_kind.IsOneOf({Lex::TokenKind::Comma, Lex::TokenKind::Semi})) {
    return false;
  }

  llvm::StringRef l = left_kind.fixed_spelling();
  llvm::StringRef r = right_kind.fixed_spelling();
  switch (r.front()) {
    case '=':
      return l == "&" || l == "^" || l == ":" || l == "=" || l == "!" ||
             l == ">" || l == ">>" || l == "<" || l == "<<" || l == "-" ||
             l == "%" || l == "|" || l == "+" || l == "/" || l == "*" ||
             l == "~";
    case '>':
      return l == "=" || l == ">" || l == "<" || l == "<=" || l == "-";
    case '?':
      return l == ":" || l == "->";
    case '-':
      return l == "-" || l == "<";
    case '+':
      return l == "+";
    case '<':
      return l == "<";
    case '/':
      return l == "/";
    case '.':
      return l == ".";
    default:
      return false;
  }
}

auto RoleForNodeKind(Parse::NodeKind kind) -> TokenRole {
  switch (kind) {
    case Parse::NodeKind::ExplicitParamListStart:
    case Parse::NodeKind::ImplicitParamListStart:
    case Parse::NodeKind::CallExprStart:
    case Parse::NodeKind::IndexExprStart:
    case Parse::NodeKind::ArrayExprOpenParen:
    case Parse::NodeKind::FormLiteralOpenParen:
    case Parse::NodeKind::TypeOfExprOpenParen:
      return TokenRole::PostfixBracket;

    case Parse::NodeKind::IdentifierNameQualifierWithParams:
    case Parse::NodeKind::IdentifierNameQualifierWithoutParams:
    case Parse::NodeKind::MemberAccessExpr:
    case Parse::NodeKind::PointerMemberAccessExpr:
      return TokenRole::MemberAccess;

    case Parse::NodeKind::ChoiceDefinitionStart:
    case Parse::NodeKind::ClassDefinitionStart:
    case Parse::NodeKind::CodeBlockStart:
    case Parse::NodeKind::FunctionDefinitionStart:
    case Parse::NodeKind::ImplDefinitionStart:
    case Parse::NodeKind::InterfaceDefinitionStart:
    case Parse::NodeKind::MatchFirstDefinitionStart:
    case Parse::NodeKind::MatchHandlerStart:
    case Parse::NodeKind::MatchStatementStart:
    case Parse::NodeKind::NamedConstraintDefinitionStart:
      return TokenRole::BlockOpen;

    case Parse::NodeKind::LambdaDefinitionStart:
      return TokenRole::LambdaOpen;

    case Parse::NodeKind::StructLiteralStart:
    case Parse::NodeKind::StructPatternStart:
    case Parse::NodeKind::StructTypeLiteralStart:
      return TokenRole::MultilineStructOpen;

    case Parse::NodeKind::ChoiceAlternativeListComma:
      return TokenRole::BlockComma;

    // A leading designator `.` (`.Red`, `{.x = 1}`, `.Self`) takes no space
    // before the name it introduces while keeping ordinary spacing before `.`.
    case Parse::NodeKind::DesignatorExpr:
    case Parse::NodeKind::StructFieldDesignator:
      return TokenRole::PrefixOperator;

#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name) \
  case Parse::NodeKind::PrefixOperator##Name:
#include "toolchain/parse/node_kind.def"
      return TokenRole::PrefixOperator;

#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name) \
  case Parse::NodeKind::PostfixOperator##Name:
#include "toolchain/parse/node_kind.def"
      return TokenRole::PostfixOperator;

    default:
      return TokenRole::Unknown;
  }
}

// Maps an infix `PrecedenceGroup` (above `Assignment`) to its split penalty by
// counting how many canonical comparable Carbon precedence tiers it binds
// strictly tighter than (`4 + looser`, producing `5`..`8`). Incomparable
// operators in Carbon require explicit parentheses, which add
// `PenaltyPerBracketDepth` per nesting level in `SolveLineBreaks`.
static auto InfixPrecedencePenalty(Parse::PrecedenceGroup group) -> int {
  static constexpr Lex::TokenKind Tiers[] = {
      Lex::TokenKind::Equal,       // Assignment
      Lex::TokenKind::And,         // Logical
      Lex::TokenKind::EqualEqual,  // Relational
      Lex::TokenKind::Plus,        // Additive / Bitwise / Shift / Modulo
  };
  int looser = 0;
  for (Lex::TokenKind tier_token : Tiers) {
    auto tier = Parse::PrecedenceGroup::ForTrailing(tier_token, /*infix=*/true);
    CARBON_CHECK(tier.has_value());
    if (group != tier->level &&
        Parse::PrecedenceGroup::GetPriority(group, tier->level) ==
            Parse::OperatorPriority::LeftFirst) {
      ++looser;
    }
  }
  return 4 + looser;
}

static auto InfixTokenPrecedencePenalty(Lex::TokenKind op_token) -> int {
  auto trailing = Parse::PrecedenceGroup::ForTrailing(op_token, /*infix=*/true);
  CARBON_CHECK(trailing.has_value() && trailing->is_binary);
  return InfixPrecedencePenalty(trailing->level);
}

auto OperatorInfoForNodeKind(Parse::NodeKind kind, const Style& style)
    -> OperatorInfo {
  // Break penalties increase with binding tightness from
  // `Parse::PrecedenceGroup` so looser operators break first.
  auto infix_info = [&style](Lex::TokenKind op_token) -> OperatorInfo {
    auto trailing =
        Parse::PrecedenceGroup::ForTrailing(op_token, /*infix=*/true);
    CARBON_CHECK(trailing.has_value() && trailing->is_binary);
    auto assign = Parse::PrecedenceGroup::ForTrailing(Lex::TokenKind::Equal,
                                                      /*infix=*/true);
    if (trailing->level == assign->level) {
      // Assignment and compound assignment continuation-indent the right-hand
      // side rather than aligning operands.
      return {.break_penalty = style.penalty_break_assignment,
              .aligns_operands = false};
    }
    return {.break_penalty = InfixPrecedencePenalty(trailing->level),
            .aligns_operands = true};
  };

  switch (kind) {
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name) \
  case Parse::NodeKind::InfixOperator##Name:        \
    return infix_info(Lex::TokenKind::Name);
#include "toolchain/parse/node_kind.def"

    case Parse::NodeKind::ShortCircuitOperatorAnd:
      return infix_info(Lex::TokenKind::And);
    case Parse::NodeKind::ShortCircuitOperatorOr:
      return infix_info(Lex::TokenKind::Or);

    // `for (... in ...)` and `where`-requirement conjunction.
    case Parse::NodeKind::ForIn:
    case Parse::NodeKind::RequirementAnd:
      return {.break_penalty = InfixTokenPrecedencePenalty(Lex::TokenKind::And),
              .aligns_operands = false};

    // Equality constraints in `where` and `observe`.
    case Parse::NodeKind::RequirementEqualEqual:
    case Parse::NodeKind::ObserveEqualEqual:
      return {.break_penalty =
                  InfixTokenPrecedencePenalty(Lex::TokenKind::EqualEqual),
              .aligns_operands = true};

    // `impls` constraints and `impl ... as` / `require ... impls` headers.
    case Parse::NodeKind::RequirementImpls:
    case Parse::NodeKind::ObserveImpls:
      return {
          .break_penalty = InfixTokenPrecedencePenalty(Lex::TokenKind::Plus),
          .aligns_operands = true};
    case Parse::NodeKind::ImplTypeAs:
    case Parse::NodeKind::ImplDefaultSelfAs:
    case Parse::NodeKind::RequireTypeImpls:
    case Parse::NodeKind::RequireDefaultSelfImpls:
      return {
          .break_penalty = InfixTokenPrecedencePenalty(Lex::TokenKind::Plus),
          .aligns_operands = false};

    // Associated-entity rewrite `.Assoc = Type` inside `where`: priced above
    // `where` and `and` so the clause wraps before splitting an individual
    // rewrite.
    case Parse::NodeKind::RequirementEqual:
      return {.break_penalty =
                  InfixTokenPrecedencePenalty(Lex::TokenKind::Star) + 1,
              .aligns_operands = false};

    default:
      return {.break_penalty = -1, .aligns_operands = false};
  }
}

auto MemberAccessBreakPenalty(Parse::NodeKind kind) -> int {
  switch (kind) {
    case Parse::NodeKind::MemberAccessExpr:
    case Parse::NodeKind::PointerMemberAccessExpr:
      // Interior member-access penalty; the formatter lowers the final link of
      // a multi-link chain to `MemberAccessLastLinkBreakPenalty`.
      return 150;
    default:
      return -1;
  }
}

// Returns true when `right` is a call, parameter-list, type-constructor, or
// subscript `(` / `[` that attaches tightly to `left` (unnamed parameter lists
// directly after `fn` or `forall` keep a separating space).
static auto IsTightPostfixBracket(const Lex::TokenizedBuffer& tokens,
                                  const TokenInfoStore& token_infos,
                                  Lex::TokenIndex left, Lex::TokenIndex right)
    -> bool {
  return token_infos.Get(right).role == TokenRole::PostfixBracket &&
         tokens.GetKind(right).IsOneOf(
             {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket}) &&
         !tokens.GetKind(left).IsOneOf(
             {Lex::TokenKind::Fn, Lex::TokenKind::Forall});
}

auto SpacesBefore(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right) -> int {
  if (WouldMergeTokens(tokens, token_infos, left, right)) {
    return 1;
  }
  Lex::TokenKind left_kind = tokens.GetKind(left);
  Lex::TokenKind right_kind = tokens.GetKind(right);
  TokenRole left_role = token_infos.Get(left).role;
  TokenRole right_role = token_infos.Get(right).role;
  if (IsTightPostfixBracket(tokens, token_infos, left, right)) {
    return 0;
  }
  if (right_role == TokenRole::PostfixOperator && right_kind.is_symbol()) {
    return 0;
  }
  if (left_role == TokenRole::PrefixOperator && left_kind.is_symbol()) {
    return 0;
  }
  if (right_kind.IsOneOf({Lex::TokenKind::Comma, Lex::TokenKind::Semi,
                          Lex::TokenKind::CloseParen,
                          Lex::TokenKind::CloseSquareBracket})) {
    return 0;
  }
  if (right_kind.IsOneOf(
          {Lex::TokenKind::Colon, Lex::TokenKind::ColonQuestion})) {
    return 0;
  }
  if (left_kind.IsOneOf(
          {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket})) {
    return 0;
  }
  if (left_role == TokenRole::StructOpen ||
      right_role == TokenRole::StructClose) {
    return 0;
  }
  if (left_kind == Lex::TokenKind::OpenCurlyBrace &&
      right_kind == Lex::TokenKind::CloseCurlyBrace) {
    return 0;
  }
  if ((left_role == TokenRole::MemberAccess &&
       left_kind.IsOneOf(
           {Lex::TokenKind::Period, Lex::TokenKind::MinusGreater})) ||
      (right_role == TokenRole::MemberAccess &&
       right_kind.IsOneOf(
           {Lex::TokenKind::Period, Lex::TokenKind::MinusGreater}))) {
    return 0;
  }
  return 1;
}

auto EscapeTripleQuoteString(llvm::StringRef text)
    -> std::optional<std::string> {
  if (!text.starts_with('#') || text.contains('\n')) {
    return std::nullopt;
  }
  size_t hash_level = 0;
  while (hash_level < text.size() && text[hash_level] == '#') {
    ++hash_level;
  }
  llvm::StringRef after_hashes = text.drop_front(hash_level);
  if (!after_hashes.starts_with("\"\"\"")) {
    return std::nullopt;
  }
  // Mirror `Lex::StringLiteral::Introducer::Lex`: `#*"""` re-lexes as a
  // multi-line introducer once followed by `\n` iff a `//` comment marker
  // (`//` followed by `' '`, `'\t'`, or EOL) hides the closing `"#*` delimiter
  // and the trimmed text before `//` contains none of `'`, `#`, or `"`.
  llvm::StringRef rest = after_hashes.drop_front(3);
  bool has_comment = false;
  for (size_t slashes = rest.find("//"); slashes != llvm::StringRef::npos;
       slashes = rest.find("//", slashes + 1)) {
    llvm::StringRef after_slashes = rest.drop_front(slashes + 2);
    if (after_slashes.empty() || after_slashes.starts_with(' ') ||
        after_slashes.starts_with('\t')) {
      rest = rest.take_front(slashes);
      has_comment = true;
      break;
    }
  }
  if (!has_comment ||
      rest.trim(" \t").find_first_of("'#\"") != llvm::StringRef::npos) {
    return std::nullopt;
  }
  llvm::StringRef hashes = text.take_front(hash_level);
  std::string out;
  out.reserve(text.size() + hash_level + 1);
  out.append(hashes.data(), hashes.size());
  out.append("\"\"\\");
  out.append(hashes.data(), hashes.size());
  out.push_back('"');
  out.append(after_hashes.data() + 3, after_hashes.size() - 3);
  return out;
}

auto CanBreakBefore(const Lex::TokenizedBuffer& tokens,
                    const TokenInfoStore& token_infos, Lex::TokenIndex left,
                    Lex::TokenIndex right) -> bool {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  Lex::TokenKind right_kind = tokens.GetKind(right);
  TokenRole right_role = token_infos.Get(right).role;
  // Never break before a separator, binding colon, closing bracket, or `=>`.
  if (right_kind.IsOneOf(
          {Lex::TokenKind::Comma, Lex::TokenKind::Semi, Lex::TokenKind::Colon,
           Lex::TokenKind::ColonQuestion, Lex::TokenKind::CloseParen,
           Lex::TokenKind::CloseSquareBracket, Lex::TokenKind::CloseCurlyBrace,
           Lex::TokenKind::EqualGreater})) {
    return false;
  }
  // Keep block, lambda, and multiline-struct `{` attached to their header,
  // and keep `else` attached to a preceding `}`. Inline `StructOpen` `{` may
  // break onto a continuation line (e.g. after `=`).
  if (right_kind == Lex::TokenKind::OpenCurlyBrace &&
      (right_role == TokenRole::BlockOpen ||
       right_role == TokenRole::LambdaOpen ||
       right_role == TokenRole::MultilineStructOpen)) {
    return false;
  }
  if (left_kind == Lex::TokenKind::CloseCurlyBrace &&
      right_kind == Lex::TokenKind::Else) {
    return false;
  }
  if (IsTightPostfixBracket(tokens, token_infos, left, right)) {
    return false;
  }
  // Never break immediately after `.`, `->`, or `->?`: member access,
  // designators, and return types break before the operator rather than after.
  if (left_kind.IsOneOf({Lex::TokenKind::Period, Lex::TokenKind::MinusGreater,
                         Lex::TokenKind::MinusGreaterQuestion})) {
    return false;
  }
  // Never separate a symbolic unary operator from its operand: Carbon's
  // fixity rules reject whitespace on the operand side.
  if (token_infos.Get(left).role == TokenRole::PrefixOperator &&
      left_kind.is_symbol()) {
    return false;
  }
  if (right_role == TokenRole::PostfixOperator) {
    return false;
  }
  // Keep `=` and infix/constraint operators attached to the preceding line and
  // break after them instead.
  if (right_kind == Lex::TokenKind::Equal ||
      token_infos.Get(right).break_penalty_after >= 0) {
    return false;
  }
  return true;
}

auto SplitPenalty(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right, const Style& style) -> int {
  Lex::TokenKind left_kind = tokens.GetKind(left);
  Lex::TokenKind right_kind = tokens.GetKind(right);
  if (left_kind == Lex::TokenKind::Comma) {
    return 1;
  }
  int left_break_penalty_after = token_infos.Get(left).break_penalty_after;
  if (left_break_penalty_after >= 0) {
    return left_break_penalty_after;
  }
  int right_break_penalty_before = token_infos.Get(right).break_penalty_before;
  if (right_break_penalty_before >= 0) {
    return right_break_penalty_before;
  }
  if (left_kind == Lex::TokenKind::Equal) {
    // Initializer or field `=` (not an `OperatorInfo` node).
    return style.penalty_break_assignment;
  }
  if (left_kind.IsOneOf(
          {Lex::TokenKind::OpenParen, Lex::TokenKind::OpenSquareBracket}) ||
      (left_kind == Lex::TokenKind::OpenCurlyBrace &&
       token_infos.Get(left).role == TokenRole::StructOpen)) {
    return style.penalty_break_before_first_call_parameter;
  }
  // After a word prefix operator (`not`, `const`, `ref`) or `...` pack
  // expansion, or before a return arrow or `unsafe` operator modifier.
  if (token_infos.Get(left).role == TokenRole::PrefixOperator ||
      left_kind == Lex::TokenKind::PeriodPeriodPeriod ||
      right_kind.IsOneOf({Lex::TokenKind::MinusGreater,
                          Lex::TokenKind::MinusGreaterQuestion,
                          Lex::TokenKind::Unsafe})) {
    return 60;
  }
  if (left_kind.IsOneOf(
          {Lex::TokenKind::Colon, Lex::TokenKind::ColonQuestion})) {
    return 100;
  }
  if (token_infos.Get(right).role == TokenRole::MemberAccess ||
      right_kind == Lex::TokenKind::Period) {
    return 150;
  }
  if (left_kind.is_keyword()) {
    // Keep introducer and modifier keywords (`private`, `fn`, `return`,
    // `while`) attached to the construct they introduce unless no cheaper break
    // exists. Word operators are already handled above.
    return 200;
  }
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

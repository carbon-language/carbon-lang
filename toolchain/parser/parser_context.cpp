// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

#include <cstdlib>
#include <memory>
#include <optional>

#include "common/check.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// A relative location for characters in errors.
enum class RelativeLocation : int8_t {
  Around,
  After,
  Before,
};

// Adapts RelativeLocation for use with formatv.
static auto operator<<(llvm::raw_ostream& out, RelativeLocation loc)
    -> llvm::raw_ostream& {
  switch (loc) {
    case RelativeLocation::Around:
      out << "around";
      break;
    case RelativeLocation::After:
      out << "after";
      break;
    case RelativeLocation::Before:
      out << "before";
      break;
  }
  return out;
}

ParserContext::ParserContext(ParseTree& tree, TokenizedBuffer& tokens,
                             TokenDiagnosticEmitter& emitter,
                             llvm::raw_ostream* vlog_stream)
    : tree_(&tree),
      tokens_(&tokens),
      emitter_(&emitter),
      vlog_stream_(vlog_stream),
      position_(tokens_->tokens().begin()),
      end_(tokens_->tokens().end()) {
  CARBON_CHECK(position_ != end_) << "Empty TokenizedBuffer";
  --end_;
  CARBON_CHECK(tokens_->GetKind(*end_) == TokenKind::EndOfFile)
      << "TokenizedBuffer should end with EndOfFile, ended with "
      << tokens_->GetKind(*end_);
}

auto ParserContext::AddLeafNode(ParseNodeKind kind,
                                TokenizedBuffer::Token token, bool has_error)
    -> void {
  tree_->node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, /*subtree_size=*/1));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto ParserContext::AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                            int subtree_start, bool has_error) -> void {
  int subtree_size = tree_->size() - subtree_start + 1;
  tree_->node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, subtree_size));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto ParserContext::ConsumeAndAddOpenParen(TokenizedBuffer::Token default_token,
                                           ParseNodeKind start_kind) -> void {
  if (auto open_paren = ConsumeIf(TokenKind::OpenParen)) {
    AddLeafNode(start_kind, *open_paren, /*has_error=*/false);
  } else {
    CARBON_DIAGNOSTIC(ExpectedParenAfter, Error, "Expected `(` after `{0}`.",
                      TokenKind);
    emitter_->Emit(*position_, ExpectedParenAfter,
                   tokens().GetKind(default_token));
    AddLeafNode(start_kind, default_token, /*has_error=*/true);
  }
}

auto ParserContext::ConsumeAndAddCloseParen(StateStackEntry state,
                                            ParseNodeKind close_kind) -> void {
  // state.token should point at the introducer, with the paren one after the
  // introducer.
  auto expected_paren = *(TokenizedBuffer::TokenIterator(state.token) + 1);

  if (tokens().GetKind(expected_paren) != TokenKind::OpenParen) {
    AddNode(close_kind, state.token, state.subtree_start, /*has_error=*/true);
  } else if (auto close_token = ConsumeIf(TokenKind::CloseParen)) {
    AddNode(close_kind, *close_token, state.subtree_start, state.has_error);
  } else {
    // TODO: Include the location of the matching open_paren in the diagnostic.
    CARBON_DIAGNOSTIC(ExpectedCloseParen, Error,
                      "Unexpected tokens before `)`.");
    emitter_->Emit(*position_, ExpectedCloseParen);

    SkipTo(tokens().GetMatchedClosingToken(expected_paren));
    AddNode(close_kind, Consume(), state.subtree_start, /*has_error=*/true);
  }
}

auto ParserContext::ConsumeAndAddLeafNodeIf(TokenKind token_kind,
                                            ParseNodeKind node_kind) -> bool {
  auto token = ConsumeIf(token_kind);
  if (!token) {
    return false;
  }

  AddLeafNode(node_kind, *token);
  return true;
}

auto ParserContext::ConsumeChecked(TokenKind kind) -> TokenizedBuffer::Token {
  CARBON_CHECK(PositionIs(kind))
      << "Required " << kind << ", found " << PositionKind();
  return Consume();
}

auto ParserContext::ConsumeIf(TokenKind kind)
    -> std::optional<TokenizedBuffer::Token> {
  if (!PositionIs(kind)) {
    return std::nullopt;
  }
  return Consume();
}

auto ParserContext::ConsumeIfPatternKeyword(TokenKind keyword_token,
                                            ParserState keyword_state,
                                            int subtree_start) -> void {
  if (auto token = ConsumeIf(keyword_token)) {
    PushState(ParserContext::StateStackEntry(
        keyword_state, PrecedenceGroup::ForTopLevelExpression(),
        PrecedenceGroup::ForTopLevelExpression(), *token, subtree_start));
  }
}

auto ParserContext::FindNextOf(std::initializer_list<TokenKind> desired_kinds)
    -> std::optional<TokenizedBuffer::Token> {
  auto new_position = position_;
  while (true) {
    TokenizedBuffer::Token token = *new_position;
    TokenKind kind = tokens().GetKind(token);
    if (kind.IsOneOf(desired_kinds)) {
      return token;
    }

    // Step to the next token at the current bracketing level.
    if (kind.is_closing_symbol() || kind == TokenKind::EndOfFile) {
      // There are no more tokens at this level.
      return std::nullopt;
    } else if (kind.is_opening_symbol()) {
      new_position = TokenizedBuffer::TokenIterator(
          tokens().GetMatchedClosingToken(token));
      // Advance past the closing token.
      ++new_position;
    } else {
      ++new_position;
    }
  }
}

auto ParserContext::SkipMatchingGroup() -> bool {
  if (!PositionKind().is_opening_symbol()) {
    return false;
  }

  SkipTo(tokens().GetMatchedClosingToken(*position_));
  ++position_;
  return true;
}

auto ParserContext::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root)
    -> std::optional<TokenizedBuffer::Token> {
  if (position_ == end_) {
    return std::nullopt;
  }

  TokenizedBuffer::Line root_line = tokens().GetLine(skip_root);
  int root_line_indent = tokens().GetIndentColumnNumber(root_line);

  // We will keep scanning through tokens on the same line as the root or
  // lines with greater indentation than root's line.
  auto is_same_line_or_indent_greater_than_root =
      [&](TokenizedBuffer::Token t) {
        TokenizedBuffer::Line l = tokens().GetLine(t);
        if (l == root_line) {
          return true;
        }

        return tokens().GetIndentColumnNumber(l) > root_line_indent;
      };

  do {
    if (PositionIs(TokenKind::CloseCurlyBrace)) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return std::nullopt;
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(TokenKind::Semi)) {
      return semi;
    }

    // Skip over any matching group of tokens().
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position_;
  } while (position_ != end_ &&
           is_same_line_or_indent_greater_than_root(*position_));

  return std::nullopt;
}

auto ParserContext::SkipTo(TokenizedBuffer::Token t) -> void {
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards from " << position_
                                << " to " << TokenizedBuffer::TokenIterator(t);
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

// Determines whether the given token is considered to be the start of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedStartOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::OpenParen, TokenKind::Identifier,
                       TokenKind::IntegerLiteral, TokenKind::RealLiteral,
                       TokenKind::StringLiteral});
}

// Determines whether the given token is considered to be the end of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedEndOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::CloseParen, TokenKind::CloseCurlyBrace,
                       TokenKind::CloseSquareBracket, TokenKind::Identifier,
                       TokenKind::IntegerLiteral, TokenKind::RealLiteral,
                       TokenKind::StringLiteral});
}

// Determines whether the given token could possibly be the start of an
// operand. This is conservatively correct, and will never incorrectly return
// `false`, but can incorrectly return `true`.
static auto IsPossibleStartOfOperand(TokenKind kind) -> bool {
  return !kind.IsOneOf({TokenKind::CloseParen, TokenKind::CloseCurlyBrace,
                        TokenKind::CloseSquareBracket, TokenKind::Comma,
                        TokenKind::Semi, TokenKind::Colon});
}

auto ParserContext::IsLexicallyValidInfixOperator() -> bool {
  CARBON_CHECK(position_ != end_) << "Expected an operator token.";

  bool leading_space = tokens().HasLeadingWhitespace(*position_);
  bool trailing_space = tokens().HasTrailingWhitespace(*position_);

  // If there's whitespace on both sides, it's an infix operator.
  if (leading_space && trailing_space) {
    return true;
  }

  // If there's whitespace on exactly one side, it's not an infix operator.
  if (leading_space || trailing_space) {
    return false;
  }

  // Otherwise, for an infix operator, the preceding token must be any close
  // bracket, identifier, or literal and the next token must be an open paren,
  // identifier, or literal.
  if (position_ == tokens().tokens().begin() ||
      !IsAssumedEndOfOperand(tokens().GetKind(*(position_ - 1))) ||
      !IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return false;
  }

  return true;
}

auto ParserContext::IsTrailingOperatorInfix() -> bool {
  if (position_ == end_) {
    return false;
  }

  // An operator that follows the infix operator rules is parsed as
  // infix, unless the next token means that it can't possibly be.
  if (IsLexicallyValidInfixOperator() &&
      IsPossibleStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return true;
  }

  // A trailing operator with leading whitespace that's not valid as infix is
  // not valid at all. If the next token looks like the start of an operand,
  // then parse as infix, otherwise as postfix. Either way we'll produce a
  // diagnostic later on.
  if (tokens().HasLeadingWhitespace(*position_) &&
      IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return true;
  }

  return false;
}

auto ParserContext::DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
  if (fixity == OperatorFixity::Infix) {
    // Infix operators must satisfy the infix operator rules.
    if (!IsLexicallyValidInfixOperator()) {
      CARBON_DIAGNOSTIC(BinaryOperatorRequiresWhitespace, Error,
                        "Whitespace missing {0} binary operator.",
                        RelativeLocation);
      emitter_->Emit(*position_, BinaryOperatorRequiresWhitespace,
                     tokens().HasLeadingWhitespace(*position_)
                         ? RelativeLocation::After
                         : (tokens().HasTrailingWhitespace(*position_)
                                ? RelativeLocation::Before
                                : RelativeLocation::Around));
    }
  } else {
    bool prefix = fixity == OperatorFixity::Prefix;

    // Whitespace is not permitted between a symbolic pre/postfix operator and
    // its operand.
    if (PositionKind().is_symbol() &&
        (prefix ? tokens().HasTrailingWhitespace(*position_)
                : tokens().HasLeadingWhitespace(*position_))) {
      CARBON_DIAGNOSTIC(UnaryOperatorHasWhitespace, Error,
                        "Whitespace is not allowed {0} this unary operator.",
                        RelativeLocation);
      emitter_->Emit(
          *position_, UnaryOperatorHasWhitespace,
          prefix ? RelativeLocation::After : RelativeLocation::Before);
    }
    // Pre/postfix operators must not satisfy the infix operator rules.
    if (IsLexicallyValidInfixOperator()) {
      CARBON_DIAGNOSTIC(UnaryOperatorRequiresWhitespace, Error,
                        "Whitespace is required {0} this unary operator.",
                        RelativeLocation);
      emitter_->Emit(
          *position_, UnaryOperatorRequiresWhitespace,
          prefix ? RelativeLocation::Before : RelativeLocation::After);
    }
  }
}

auto ParserContext::ConsumeListToken(ParseNodeKind comma_kind,
                                     TokenKind close_kind,
                                     bool already_has_error) -> ListTokenKind {
  if (!PositionIs(TokenKind::Comma) && !PositionIs(close_kind)) {
    // Don't error a second time on the same element.
    if (!already_has_error) {
      CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                        "Expected `,` or `{0}`.", TokenKind);
      emitter_->Emit(*position_, UnexpectedTokenAfterListElement, close_kind);
      ReturnErrorOnState();
    }

    // Recover from the invalid token.
    auto end_of_element = FindNextOf({TokenKind::Comma, close_kind});
    // The lexer guarantees that parentheses are balanced.
    CARBON_CHECK(end_of_element)
        << "missing matching `" << close_kind.opening_symbol() << "` for `"
        << close_kind << "`";

    SkipTo(*end_of_element);
  }

  if (PositionIs(close_kind)) {
    return ListTokenKind::Close;
  } else {
    AddLeafNode(comma_kind, Consume());
    return PositionIs(close_kind) ? ListTokenKind::CommaClose
                                  : ListTokenKind::Comma;
  }
}

auto ParserContext::GetDeclarationContext() -> DeclarationContext {
  // i == 0 is the file-level DeclarationScopeLoop. Additionally, i == 1 can be
  // skipped because it will never be a DeclarationScopeLoop.
  for (int i = state_stack_.size() - 1; i > 1; --i) {
    // The declaration context is always the state _above_ a
    // DeclarationScopeLoop.
    if (state_stack_[i].state == ParserState::DeclarationScopeLoop) {
      switch (state_stack_[i - 1].state) {
        case ParserState::TypeDefinitionFinishAsClass:
          return DeclarationContext::Class;
        case ParserState::TypeDefinitionFinishAsInterface:
          return DeclarationContext::Interface;
        case ParserState::TypeDefinitionFinishAsNamedConstraint:
          return DeclarationContext::NamedConstraint;
        default:
          llvm_unreachable("Missing handling for a declaration scope");
      }
    }
  }
  CARBON_CHECK(!state_stack_.empty() &&
               state_stack_[0].state == ParserState::DeclarationScopeLoop);
  return DeclarationContext::File;
}

auto ParserContext::RecoverFromDeclarationError(StateStackEntry state,
                                                ParseNodeKind parse_node_kind,
                                                bool skip_past_likely_end)
    -> void {
  auto token = state.token;
  if (skip_past_likely_end) {
    if (auto semi = SkipPastLikelyEnd(token)) {
      token = *semi;
    }
  }
  AddNode(parse_node_kind, token, state.subtree_start,
          /*has_error=*/true);
}

}  // namespace Carbon

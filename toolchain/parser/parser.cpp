// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser.h"

#include <cstdlib>
#include <memory>
#include <optional>

#include "common/check.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// May be emitted a couple different ways as part of operator parsing.
CARBON_DIAGNOSTIC(
    OperatorRequiresParentheses, Error,
    "Parentheses are required to disambiguate operator precedence.");

CARBON_DIAGNOSTIC(ExpectedSemiAfterExpression, Error,
                  "Expected `;` after expression.");

CARBON_DIAGNOSTIC(ExpectedDeclarationName, Error,
                  "`{0}` introducer should be followed by a name.", TokenKind);
CARBON_DIAGNOSTIC(ExpectedDeclarationSemiOrDefinition, Error,
                  "`{0}` should either end with a `;` for a declaration or "
                  "have a `{{ ... }` block for a definition.",
                  TokenKind);

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

class Parser::PrettyStackTraceParseState : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceParseState(const Parser* parser) : parser_(parser) {}
  ~PrettyStackTraceParseState() override = default;

  auto print(llvm::raw_ostream& output) const -> void override {
    output << "Parser stack:\n";
    for (int i = 0; i < static_cast<int>(parser_->state_stack_.size()); ++i) {
      const auto& entry = parser_->state_stack_[i];
      output << "\t" << i << ".\t" << entry.state;
      Print(output, entry.token);
    }
    output << "\tcursor\tposition_";
    Print(output, *parser_->position_);
  }

 private:
  auto Print(llvm::raw_ostream& output, TokenizedBuffer::Token token) const
      -> void {
    auto line = parser_->tokens_->GetLine(token);
    output << " @ " << parser_->tokens_->GetLineNumber(line) << ":"
           << parser_->tokens_->GetColumnNumber(token) << ":"
           << " token " << token << " : " << parser_->tokens_->GetKind(token)
           << "\n";
  }

  const Parser* parser_;
};

Parser::Parser(ParseTree& tree, TokenizedBuffer& tokens,
               TokenDiagnosticEmitter& emitter, llvm::raw_ostream* vlog_stream)
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

auto Parser::AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                         bool has_error) -> void {
  tree_->node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, /*subtree_size=*/1));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto Parser::AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                     int subtree_start, bool has_error) -> void {
  int subtree_size = tree_->size() - subtree_start + 1;
  tree_->node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, subtree_size));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto Parser::ConsumeAndAddOpenParen(TokenizedBuffer::Token default_token,
                                    ParseNodeKind start_kind) -> void {
  if (auto open_paren = ConsumeIf(TokenKind::OpenParen)) {
    AddLeafNode(start_kind, *open_paren, /*has_error=*/false);
  } else {
    CARBON_DIAGNOSTIC(ExpectedParenAfter, Error, "Expected `(` after `{0}`.",
                      TokenKind);
    emitter_->Emit(*position_, ExpectedParenAfter,
                   tokens_->GetKind(default_token));
    AddLeafNode(start_kind, default_token, /*has_error=*/true);
  }
}

auto Parser::ConsumeAndAddCloseParen(StateStackEntry state,
                                     ParseNodeKind close_kind) -> void {
  // state.token should point at the introducer, with the paren one after the
  // introducer.
  auto expected_paren = *(TokenizedBuffer::TokenIterator(state.token) + 1);

  if (tokens_->GetKind(expected_paren) != TokenKind::OpenParen) {
    AddNode(close_kind, state.token, state.subtree_start, /*has_error=*/true);
  } else if (auto close_token = ConsumeIf(TokenKind::CloseParen)) {
    AddNode(close_kind, *close_token, state.subtree_start, state.has_error);
  } else {
    // TODO: Include the location of the matching open_paren in the diagnostic.
    CARBON_DIAGNOSTIC(ExpectedCloseParen, Error,
                      "Unexpected tokens before `)`.");
    emitter_->Emit(*position_, ExpectedCloseParen);

    SkipTo(tokens_->GetMatchedClosingToken(expected_paren));
    AddNode(close_kind, Consume(), state.subtree_start, /*has_error=*/true);
  }
}

auto Parser::ConsumeAndAddLeafNodeIf(TokenKind token_kind,
                                     ParseNodeKind node_kind) -> bool {
  auto token = ConsumeIf(token_kind);
  if (!token) {
    return false;
  }

  AddLeafNode(node_kind, *token);
  return true;
}

auto Parser::ConsumeChecked(TokenKind kind) -> TokenizedBuffer::Token {
  CARBON_CHECK(PositionIs(kind))
      << "Required " << kind << ", found " << PositionKind();
  return Consume();
}

auto Parser::ConsumeIf(TokenKind kind)
    -> std::optional<TokenizedBuffer::Token> {
  if (!PositionIs(kind)) {
    return std::nullopt;
  }
  return Consume();
}

auto Parser::FindNextOf(std::initializer_list<TokenKind> desired_kinds)
    -> std::optional<TokenizedBuffer::Token> {
  auto new_position = position_;
  while (true) {
    TokenizedBuffer::Token token = *new_position;
    TokenKind kind = tokens_->GetKind(token);
    if (kind.IsOneOf(desired_kinds)) {
      return token;
    }

    // Step to the next token at the current bracketing level.
    if (kind.is_closing_symbol() || kind == TokenKind::EndOfFile) {
      // There are no more tokens at this level.
      return std::nullopt;
    } else if (kind.is_opening_symbol()) {
      new_position = TokenizedBuffer::TokenIterator(
          tokens_->GetMatchedClosingToken(token));
      // Advance past the closing token.
      ++new_position;
    } else {
      ++new_position;
    }
  }
}

auto Parser::SkipMatchingGroup() -> bool {
  if (!PositionKind().is_opening_symbol()) {
    return false;
  }

  SkipTo(tokens_->GetMatchedClosingToken(*position_));
  ++position_;
  return true;
}

auto Parser::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root)
    -> std::optional<TokenizedBuffer::Token> {
  if (position_ == end_) {
    return std::nullopt;
  }

  TokenizedBuffer::Line root_line = tokens_->GetLine(skip_root);
  int root_line_indent = tokens_->GetIndentColumnNumber(root_line);

  // We will keep scanning through tokens on the same line as the root or
  // lines with greater indentation than root's line.
  auto is_same_line_or_indent_greater_than_root =
      [&](TokenizedBuffer::Token t) {
        TokenizedBuffer::Line l = tokens_->GetLine(t);
        if (l == root_line) {
          return true;
        }

        return tokens_->GetIndentColumnNumber(l) > root_line_indent;
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

    // Skip over any matching group of tokens_->
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position_;
  } while (position_ != end_ &&
           is_same_line_or_indent_greater_than_root(*position_));

  return std::nullopt;
}

auto Parser::SkipTo(TokenizedBuffer::Token t) -> void {
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards from " << position_
                                << " to " << TokenizedBuffer::TokenIterator(t);
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

auto Parser::HandleCodeBlockState() -> void {
  PopAndDiscardState();

  PushState(ParserState::CodeBlockFinish);
  if (ConsumeAndAddLeafNodeIf(TokenKind::OpenCurlyBrace,
                              ParseNodeKind::CodeBlockStart)) {
    PushState(ParserState::StatementScopeLoop);
  } else {
    AddLeafNode(ParseNodeKind::CodeBlockStart, *position_,
                /*has_error=*/true);

    // Recover by parsing a single statement.
    CARBON_DIAGNOSTIC(ExpectedCodeBlock, Error, "Expected braced code block.");
    emitter_->Emit(*position_, ExpectedCodeBlock);

    PushState(ParserState::Statement);
  }
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

auto Parser::IsLexicallyValidInfixOperator() -> bool {
  CARBON_CHECK(position_ != end_) << "Expected an operator token.";

  bool leading_space = tokens_->HasLeadingWhitespace(*position_);
  bool trailing_space = tokens_->HasTrailingWhitespace(*position_);

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
  if (position_ == tokens_->tokens().begin() ||
      !IsAssumedEndOfOperand(tokens_->GetKind(*(position_ - 1))) ||
      !IsAssumedStartOfOperand(tokens_->GetKind(*(position_ + 1)))) {
    return false;
  }

  return true;
}

auto Parser::IsTrailingOperatorInfix() -> bool {
  if (position_ == end_) {
    return false;
  }

  // An operator that follows the infix operator rules is parsed as
  // infix, unless the next token means that it can't possibly be.
  if (IsLexicallyValidInfixOperator() &&
      IsPossibleStartOfOperand(tokens_->GetKind(*(position_ + 1)))) {
    return true;
  }

  // A trailing operator with leading whitespace that's not valid as infix is
  // not valid at all. If the next token looks like the start of an operand,
  // then parse as infix, otherwise as postfix. Either way we'll produce a
  // diagnostic later on.
  if (tokens_->HasLeadingWhitespace(*position_) &&
      IsAssumedStartOfOperand(tokens_->GetKind(*(position_ + 1)))) {
    return true;
  }

  return false;
}

auto Parser::DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
  if (fixity == OperatorFixity::Infix) {
    // Infix operators must satisfy the infix operator rules.
    if (!IsLexicallyValidInfixOperator()) {
      CARBON_DIAGNOSTIC(BinaryOperatorRequiresWhitespace, Error,
                        "Whitespace missing {0} binary operator.",
                        RelativeLocation);
      emitter_->Emit(*position_, BinaryOperatorRequiresWhitespace,
                     tokens_->HasLeadingWhitespace(*position_)
                         ? RelativeLocation::After
                         : (tokens_->HasTrailingWhitespace(*position_)
                                ? RelativeLocation::Before
                                : RelativeLocation::Around));
    }
  } else {
    bool prefix = fixity == OperatorFixity::Prefix;

    // Whitespace is not permitted between a symbolic pre/postfix operator and
    // its operand.
    if (PositionKind().is_symbol() &&
        (prefix ? tokens_->HasTrailingWhitespace(*position_)
                : tokens_->HasLeadingWhitespace(*position_))) {
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

auto Parser::ConsumeListToken(ParseNodeKind comma_kind, TokenKind close_kind,
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

auto Parser::Parse() -> void {
  // Traces state_stack_. This runs even in opt because it's low overhead.
  PrettyStackTraceParseState pretty_stack(this);

  PushState(ParserState::DeclarationScopeLoop);

  // The package should always be the first token, if it's present. Any other
  // use is invalid.
  if (PositionIs(TokenKind::Package)) {
    PushState(ParserState::Package);
  }

  while (!state_stack_.empty()) {
    switch (state_stack_.back().state) {
#define CARBON_PARSER_STATE(Name) \
  case ParserState::Name:         \
    Handle##Name##State();        \
    break;
#include "toolchain/parser/parser_state.def"
    }
  }

  AddLeafNode(ParseNodeKind::FileEnd, *position_);
}

auto Parser::GetDeclarationContext() -> DeclarationContext {
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

auto Parser::HandleDeclarationError(StateStackEntry state,
                                    ParseNodeKind parse_node_kind,
                                    bool skip_past_likely_end) -> void {
  auto token = state.token;
  if (skip_past_likely_end) {
    if (auto semi = SkipPastLikelyEnd(token)) {
      token = *semi;
    }
  }
  AddNode(parse_node_kind, token, state.subtree_start,
          /*has_error=*/true);
}

auto Parser::HandleUnrecognizedDeclaration() -> void {
  CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                    "Unrecognized declaration introducer.");
  emitter_->Emit(*position_, UnrecognizedDeclaration);
  auto cursor = *position_;
  auto semi = SkipPastLikelyEnd(cursor);
  // Locate the EmptyDeclaration at the semi when found, but use the
  // original cursor location for an error when not.
  AddLeafNode(ParseNodeKind::EmptyDeclaration, semi ? *semi : cursor,
              /*has_error=*/true);
}

auto Parser::HandleBraceExpressionState() -> void {
  auto state = PopState();

  state.state = ParserState::BraceExpressionFinishAsUnknown;
  PushState(state);

  CARBON_CHECK(ConsumeAndAddLeafNodeIf(
      TokenKind::OpenCurlyBrace,
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart));
  if (!PositionIs(TokenKind::CloseCurlyBrace)) {
    PushState(ParserState::BraceExpressionParameterAsUnknown);
  }
}

auto Parser::HandleBraceExpressionParameterError(StateStackEntry state,
                                                 ParserState param_finish_state)
    -> void {
  bool is_type =
      param_finish_state == ParserState::BraceExpressionParameterFinishAsType;
  bool is_value =
      param_finish_state == ParserState::BraceExpressionParameterFinishAsValue;
  bool is_unknown = param_finish_state ==
                    ParserState::BraceExpressionParameterFinishAsUnknown;
  CARBON_CHECK(is_type || is_value || is_unknown);
  CARBON_DIAGNOSTIC(ExpectedStructLiteralField, Error, "Expected {0}{1}{2}.",
                    llvm::StringRef, llvm::StringRef, llvm::StringRef);
  emitter_->Emit(*position_, ExpectedStructLiteralField,
                 (is_type || is_unknown) ? "`.field: field_type`" : "",
                 is_unknown ? " or " : "",
                 (is_value || is_unknown) ? "`.field = value`" : "");

  state.state = param_finish_state;
  state.has_error = true;
  PushState(state);
}

auto Parser::HandleBraceExpressionParameter(ParserState after_designator_state,
                                            ParserState param_finish_state)
    -> void {
  auto state = PopState();

  if (!PositionIs(TokenKind::Period)) {
    HandleBraceExpressionParameterError(state, param_finish_state);
    return;
  }

  state.state = after_designator_state;
  PushState(state);
  PushState(ParserState::DesignatorAsStruct);
}

auto Parser::HandleBraceExpressionParameterAsTypeState() -> void {
  HandleBraceExpressionParameter(
      ParserState::BraceExpressionParameterAfterDesignatorAsType,
      ParserState::BraceExpressionParameterFinishAsType);
}

auto Parser::HandleBraceExpressionParameterAsValueState() -> void {
  HandleBraceExpressionParameter(
      ParserState::BraceExpressionParameterAfterDesignatorAsValue,
      ParserState::BraceExpressionParameterFinishAsValue);
}

auto Parser::HandleBraceExpressionParameterAsUnknownState() -> void {
  HandleBraceExpressionParameter(
      ParserState::BraceExpressionParameterAfterDesignatorAsUnknown,
      ParserState::BraceExpressionParameterFinishAsUnknown);
}

auto Parser::HandleBraceExpressionParameterAfterDesignator(
    ParserState param_finish_state) -> void {
  auto state = PopState();

  if (state.has_error) {
    auto recovery_pos =
        FindNextOf({TokenKind::Equal, TokenKind::Colon, TokenKind::Comma});
    if (!recovery_pos || tokens_->GetKind(*recovery_pos) == TokenKind::Comma) {
      state.state = param_finish_state;
      PushState(state);
      return;
    }
    SkipTo(*recovery_pos);
  }

  // Work out the kind of this element.
  bool is_type;
  if (PositionIs(TokenKind::Colon)) {
    is_type = true;
  } else if (PositionIs(TokenKind::Equal)) {
    is_type = false;
  } else {
    HandleBraceExpressionParameterError(state, param_finish_state);
    return;
  }

  // If we're changing from unknown, update the related finish states.
  if (param_finish_state ==
      ParserState::BraceExpressionParameterFinishAsUnknown) {
    auto finish_state = PopState();
    CARBON_CHECK(finish_state.state ==
                 ParserState::BraceExpressionFinishAsUnknown);
    if (is_type) {
      finish_state.state = ParserState::BraceExpressionFinishAsType;
      param_finish_state = ParserState::BraceExpressionParameterFinishAsType;
    } else {
      finish_state.state = ParserState::BraceExpressionFinishAsValue;
      param_finish_state = ParserState::BraceExpressionParameterFinishAsValue;
    }
    PushState(finish_state);
  }

  auto want_param_finish_state =
      is_type ? ParserState::BraceExpressionParameterFinishAsType
              : ParserState::BraceExpressionParameterFinishAsValue;
  if (param_finish_state != want_param_finish_state) {
    HandleBraceExpressionParameterError(state, param_finish_state);
    return;
  }

  // Struct type fields and value fields use the same grammar except
  // that one has a `:` separator and the other has an `=` separator.
  state.state = param_finish_state;
  state.token = Consume();
  PushState(state);
  PushState(ParserState::Expression);
}

auto Parser::HandleBraceExpressionParameterAfterDesignatorAsTypeState()
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      ParserState::BraceExpressionParameterFinishAsType);
}

auto Parser::HandleBraceExpressionParameterAfterDesignatorAsValueState()
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      ParserState::BraceExpressionParameterFinishAsValue);
}

auto Parser::HandleBraceExpressionParameterAfterDesignatorAsUnknownState()
    -> void {
  HandleBraceExpressionParameterAfterDesignator(
      ParserState::BraceExpressionParameterFinishAsUnknown);
}

auto Parser::HandleBraceExpressionParameterFinish(ParseNodeKind node_kind,
                                                  ParserState param_state)
    -> void {
  auto state = PopState();

  if (state.has_error) {
    AddLeafNode(ParseNodeKind::StructFieldUnknown, state.token,
                /*has_error=*/true);
  } else {
    AddNode(node_kind, state.token, state.subtree_start, /*has_error=*/false);
  }

  if (ConsumeListToken(ParseNodeKind::StructComma, TokenKind::CloseCurlyBrace,
                       state.has_error) == ListTokenKind::Comma) {
    PushState(param_state);
  }
}

auto Parser::HandleBraceExpressionParameterFinishAsTypeState() -> void {
  HandleBraceExpressionParameterFinish(
      ParseNodeKind::StructFieldType,
      ParserState::BraceExpressionParameterAsType);
}

auto Parser::HandleBraceExpressionParameterFinishAsValueState() -> void {
  HandleBraceExpressionParameterFinish(
      ParseNodeKind::StructFieldValue,
      ParserState::BraceExpressionParameterAsValue);
}

auto Parser::HandleBraceExpressionParameterFinishAsUnknownState() -> void {
  HandleBraceExpressionParameterFinish(
      ParseNodeKind::StructFieldUnknown,
      ParserState::BraceExpressionParameterAsUnknown);
}

auto Parser::HandleBraceExpressionFinish(ParseNodeKind node_kind) -> void {
  auto state = PopState();

  AddNode(node_kind, Consume(), state.subtree_start, state.has_error);
}

auto Parser::HandleBraceExpressionFinishAsTypeState() -> void {
  HandleBraceExpressionFinish(ParseNodeKind::StructTypeLiteral);
}

auto Parser::HandleBraceExpressionFinishAsValueState() -> void {
  HandleBraceExpressionFinish(ParseNodeKind::StructLiteral);
}

auto Parser::HandleBraceExpressionFinishAsUnknownState() -> void {
  HandleBraceExpressionFinish(ParseNodeKind::StructLiteral);
}

auto Parser::HandleCallExpressionState() -> void {
  auto state = PopState();

  state.state = ParserState::CallExpressionFinish;
  PushState(state);

  AddNode(ParseNodeKind::CallExpressionStart, Consume(), state.subtree_start,
          state.has_error);
  if (!PositionIs(TokenKind::CloseParen)) {
    PushState(ParserState::CallExpressionParameterFinish);
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleCallExpressionParameterFinishState() -> void {
  auto state = PopState();

  if (state.has_error) {
    ReturnErrorOnState();
  }

  if (ConsumeListToken(ParseNodeKind::CallExpressionComma,
                       TokenKind::CloseParen,
                       state.has_error) == ListTokenKind::Comma) {
    PushState(ParserState::CallExpressionParameterFinish);
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleCallExpressionFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::CallExpression, Consume(), state.subtree_start,
          state.has_error);
}

auto Parser::HandleCodeBlockFinishState() -> void {
  auto state = PopState();

  // If the block started with an open curly, this is a close curly.
  if (tokens_->GetKind(state.token) == TokenKind::OpenCurlyBrace) {
    AddNode(ParseNodeKind::CodeBlock, Consume(), state.subtree_start,
            state.has_error);
  } else {
    AddNode(ParseNodeKind::CodeBlock, state.token, state.subtree_start,
            /*has_error=*/true);
  }
}

auto Parser::HandleDeclarationNameAndParams(bool params_required) -> void {
  auto state = PopState();

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                               ParseNodeKind::DeclaredName)) {
    emitter_->Emit(*position_, ExpectedDeclarationName,
                   tokens_->GetKind(state.token));
    ReturnErrorOnState();
    return;
  }

  if (PositionIs(TokenKind::OpenSquareBracket)) {
    PushState(ParserState::DeclarationNameAndParamsAfterDeduced);
    PushState(ParserState::ParameterListAsDeduced);
  } else if (PositionIs(TokenKind::OpenParen)) {
    PushState(ParserState::ParameterListAsRegular);
  } else if (params_required) {
    CARBON_DIAGNOSTIC(ParametersRequiredByIntroducer, Error,
                      "`{0}` requires a `(` for parameters.", TokenKind);
    emitter_->Emit(*position_, ParametersRequiredByIntroducer,
                   tokens_->GetKind(state.token));
    ReturnErrorOnState();
  }
}

auto Parser::HandleDeclarationNameAndParamsAsOptionalState() -> void {
  HandleDeclarationNameAndParams(/*params_required=*/false);
}

auto Parser::HandleDeclarationNameAndParamsAsRequiredState() -> void {
  HandleDeclarationNameAndParams(/*params_required=*/true);
}

auto Parser::HandleDeclarationNameAndParamsAfterDeducedState() -> void {
  PopAndDiscardState();

  if (PositionIs(TokenKind::OpenParen)) {
    PushState(ParserState::ParameterListAsRegular);
  } else {
    CARBON_DIAGNOSTIC(
        ParametersRequiredByDeduced, Error,
        "A `(` for parameters is required after deduced parameters.");
    emitter_->Emit(*position_, ParametersRequiredByDeduced);
    ReturnErrorOnState();
  }
}

auto Parser::HandleDeclarationScopeLoopState() -> void {
  // This maintains the current state unless we're at the end of the scope.

  switch (PositionKind()) {
    case TokenKind::CloseCurlyBrace:
    case TokenKind::EndOfFile: {
      // This is the end of the scope, so the loop state ends.
      PopAndDiscardState();
      break;
    }
    case TokenKind::Class: {
      PushState(ParserState::TypeIntroducerAsClass);
      break;
    }
    case TokenKind::Constraint: {
      PushState(ParserState::TypeIntroducerAsNamedConstraint);
      break;
    }
    case TokenKind::Fn: {
      PushState(ParserState::FunctionIntroducer);
      break;
    }
    case TokenKind::Interface: {
      PushState(ParserState::TypeIntroducerAsInterface);
      break;
    }
    case TokenKind::Semi: {
      AddLeafNode(ParseNodeKind::EmptyDeclaration, Consume());
      break;
    }
    case TokenKind::Var: {
      PushState(ParserState::VarAsSemicolon);
      break;
    }
    default: {
      HandleUnrecognizedDeclaration();
      break;
    }
  }
}

auto Parser::HandleDesignator(bool as_struct) -> void {
  auto state = PopState();

  // `.` identifier
  auto dot = ConsumeChecked(TokenKind::Period);

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                               ParseNodeKind::DesignatedName)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterDot, Error,
                      "Expected identifier after `.`.");
    emitter_->Emit(*position_, ExpectedIdentifierAfterDot);
    // If we see a keyword, assume it was intended to be the designated name.
    // TODO: Should keywords be valid in designators?
    if (PositionKind().is_keyword()) {
      AddLeafNode(ParseNodeKind::DesignatedName, Consume(),
                  /*has_error=*/true);
    } else {
      AddLeafNode(ParseNodeKind::DesignatedName, *position_,
                  /*has_error=*/true);
      // Indicate the error to the parent state so that it can avoid producing
      // more errors.
      ReturnErrorOnState();
    }
  }

  AddNode(as_struct ? ParseNodeKind::StructFieldDesignator
                    : ParseNodeKind::DesignatorExpression,
          dot, state.subtree_start, state.has_error);
}

auto Parser::HandleDesignatorAsExpressionState() -> void {
  HandleDesignator(/*as_struct=*/false);
}

auto Parser::HandleDesignatorAsStructState() -> void {
  HandleDesignator(/*as_struct=*/true);
}

auto Parser::HandleExpressionState() -> void {
  auto state = PopState();

  // Check for a prefix operator.
  if (auto operator_precedence = PrecedenceGroup::ForLeading(PositionKind())) {
    if (PrecedenceGroup::GetPriority(state.ambient_precedence,
                                     *operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this prefix operator in this
      // context. Diagnose this, but carry on and parse it anyway.
      emitter_->Emit(*position_, OperatorRequiresParentheses);
    } else {
      // Check that this operator follows the proper whitespace rules.
      DiagnoseOperatorFixity(OperatorFixity::Prefix);
    }

    PushStateForExpressionLoop(ParserState::ExpressionLoopForPrefix,
                               state.ambient_precedence, *operator_precedence);
    ++position_;
    PushStateForExpression(*operator_precedence);
  } else {
    PushStateForExpressionLoop(ParserState::ExpressionLoop,
                               state.ambient_precedence,
                               PrecedenceGroup::ForPostfixExpression());
    PushState(ParserState::ExpressionInPostfix);
  }
}

auto Parser::HandleExpressionInPostfixState() -> void {
  auto state = PopState();

  // Continue to the loop state.
  state.state = ParserState::ExpressionInPostfixLoop;

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  switch (PositionKind()) {
    case TokenKind::Identifier: {
      AddLeafNode(ParseNodeKind::NameReference, Consume());
      PushState(state);
      break;
    }
    case TokenKind::IntegerLiteral:
    case TokenKind::RealLiteral:
    case TokenKind::StringLiteral:
    case TokenKind::IntegerTypeLiteral:
    case TokenKind::UnsignedIntegerTypeLiteral:
    case TokenKind::FloatingPointTypeLiteral:
    case TokenKind::StringTypeLiteral: {
      AddLeafNode(ParseNodeKind::Literal, Consume());
      PushState(state);
      break;
    }
    case TokenKind::OpenCurlyBrace: {
      PushState(state);
      PushState(ParserState::BraceExpression);
      break;
    }
    case TokenKind::OpenParen: {
      PushState(state);
      PushState(ParserState::ParenExpression);
      break;
    }
    case TokenKind::SelfValueIdentifier: {
      AddLeafNode(ParseNodeKind::SelfValueIdentifier, Consume());
      PushState(state);
      break;
    }
    case TokenKind::SelfTypeIdentifier: {
      AddLeafNode(ParseNodeKind::SelfTypeIdentifier, Consume());
      PushState(state);
      break;
    }
    default: {
      // Add a node to keep the parse tree balanced.
      AddLeafNode(ParseNodeKind::InvalidParse, *position_, /*has_error=*/true);
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      emitter_->Emit(*position_, ExpectedExpression);
      ReturnErrorOnState();
      break;
    }
  }
}

auto Parser::HandleExpressionInPostfixLoopState() -> void {
  // This is a cyclic state that repeats, so this state is typically pushed back
  // on.
  auto state = PopState();

  state.token = *position_;

  switch (PositionKind()) {
    case TokenKind::Period: {
      PushState(state);
      state.state = ParserState::DesignatorAsExpression;
      PushState(state);
      break;
    }
    case TokenKind::OpenParen: {
      PushState(state);
      state.state = ParserState::CallExpression;
      PushState(state);
      break;
    }
    default: {
      if (state.has_error) {
        ReturnErrorOnState();
      }
      break;
    }
  }
}

auto Parser::HandleExpressionLoopState() -> void {
  auto state = PopState();

  auto trailing_operator =
      PrecedenceGroup::ForTrailing(PositionKind(), IsTrailingOperatorInfix());
  if (!trailing_operator) {
    if (state.has_error) {
      ReturnErrorOnState();
    }
    return;
  }
  auto [operator_precedence, is_binary] = *trailing_operator;

  // TODO: If this operator is ambiguous with either the ambient precedence
  // or the LHS precedence, and there's a variant with a different fixity
  // that would work, use that one instead for error recovery.
  if (PrecedenceGroup::GetPriority(state.ambient_precedence,
                                   operator_precedence) !=
      OperatorPriority::RightFirst) {
    // The precedence rules don't permit this operator in this context. Try
    // again in the enclosing expression context.
    if (state.has_error) {
      ReturnErrorOnState();
    }
    return;
  }

  if (PrecedenceGroup::GetPriority(state.lhs_precedence, operator_precedence) !=
      OperatorPriority::LeftFirst) {
    // Either the LHS operator and this operator are ambiguous, or the
    // LHS operator is a unary operator that can't be nested within
    // this operator. Either way, parentheses are required.
    emitter_->Emit(*position_, OperatorRequiresParentheses);
    state.has_error = true;
  } else {
    DiagnoseOperatorFixity(is_binary ? OperatorFixity::Infix
                                     : OperatorFixity::Postfix);
  }

  state.token = Consume();
  state.lhs_precedence = operator_precedence;

  if (is_binary) {
    state.state = ParserState::ExpressionLoopForBinary;
    PushState(state);
    PushStateForExpression(operator_precedence);
  } else {
    AddNode(ParseNodeKind::PostfixOperator, state.token, state.subtree_start,
            state.has_error);
    state.has_error = false;
    PushState(state);
  }
}

auto Parser::HandleExpressionLoopForBinaryState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::InfixOperator, state.token, state.subtree_start,
          state.has_error);
  state.state = ParserState::ExpressionLoop;
  state.has_error = false;
  PushState(state);
}

auto Parser::HandleExpressionLoopForPrefixState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::PrefixOperator, state.token, state.subtree_start,
          state.has_error);
  state.state = ParserState::ExpressionLoop;
  state.has_error = false;
  PushState(state);
}

auto Parser::HandleExpressionStatementFinishState() -> void {
  auto state = PopState();

  if (auto semi = ConsumeIf(TokenKind::Semi)) {
    AddNode(ParseNodeKind::ExpressionStatement, *semi, state.subtree_start,
            state.has_error);
    return;
  }

  if (!state.has_error) {
    emitter_->Emit(*position_, ExpectedSemiAfterExpression);
  }

  if (auto semi_token = SkipPastLikelyEnd(state.token)) {
    AddNode(ParseNodeKind::ExpressionStatement, *semi_token,
            state.subtree_start,
            /*has_error=*/true);
    return;
  }

  // Found junk not even followed by a `;`, no node to add.
  ReturnErrorOnState();
}

auto Parser::HandleFunctionIntroducerState() -> void {
  auto state = PopState();

  AddLeafNode(ParseNodeKind::FunctionIntroducer, Consume());

  state.state = ParserState::FunctionAfterParameters;
  PushState(state);
  state.state = ParserState::DeclarationNameAndParamsAsRequired;
  PushState(state);
}

auto Parser::HandleFunctionAfterParametersState() -> void {
  auto state = PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  state.state = ParserState::FunctionSignatureFinish;
  PushState(state);

  // If there is a return type, parse the expression before adding the return
  // type nod.e
  if (PositionIs(TokenKind::MinusGreater)) {
    PushState(ParserState::FunctionReturnTypeFinish);
    ++position_;
    PushStateForExpression(PrecedenceGroup::ForType());
  }
}

auto Parser::HandleFunctionReturnTypeFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::ReturnType, state.token, state.subtree_start,
          state.has_error);
}

auto Parser::HandleFunctionSignatureFinishState() -> void {
  auto state = PopState();

  switch (PositionKind()) {
    case TokenKind::Semi: {
      AddNode(ParseNodeKind::FunctionDeclaration, Consume(),
              state.subtree_start, state.has_error);
      break;
    }
    case TokenKind::OpenCurlyBrace: {
      if (auto context = GetDeclarationContext();
          context == DeclarationContext::Interface ||
          context == DeclarationContext::NamedConstraint) {
        CARBON_DIAGNOSTIC(
            MethodImplNotAllowed, Error,
            "Method implementations are not allowed in interfaces.");
        emitter_->Emit(*position_, MethodImplNotAllowed);
        HandleDeclarationError(state, ParseNodeKind::FunctionDeclaration,
                               /*skip_past_likely_end=*/true);
        break;
      }

      AddNode(ParseNodeKind::FunctionDefinitionStart, Consume(),
              state.subtree_start, state.has_error);
      // Any error is recorded on the FunctionDefinitionStart.
      state.has_error = false;
      state.state = ParserState::FunctionDefinitionFinish;
      PushState(state);
      PushState(ParserState::StatementScopeLoop);
      break;
    }
    default: {
      if (!state.has_error) {
        emitter_->Emit(*position_, ExpectedDeclarationSemiOrDefinition,
                       TokenKind::Fn);
      }
      // Only need to skip if we've not already found a new line.
      bool skip_past_likely_end =
          tokens_->GetLine(*position_) == tokens_->GetLine(state.token);
      HandleDeclarationError(state, ParseNodeKind::FunctionDeclaration,
                             skip_past_likely_end);
      break;
    }
  }
}

auto Parser::HandleFunctionDefinitionFinishState() -> void {
  auto state = PopState();
  AddNode(ParseNodeKind::FunctionDefinition, Consume(), state.subtree_start,
          state.has_error);
}

auto Parser::HandlePackageState() -> void {
  auto state = PopState();

  AddLeafNode(ParseNodeKind::PackageIntroducer, Consume());

  auto exit_on_parse_error = [&]() {
    auto semi_token = SkipPastLikelyEnd(state.token);
    return AddNode(ParseNodeKind::PackageDirective,
                   semi_token ? *semi_token : state.token, state.subtree_start,
                   /*has_error=*/true);
  };

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier,
                               ParseNodeKind::DeclaredName)) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPackage, Error,
                      "Expected identifier after `package`.");
    emitter_->Emit(*position_, ExpectedIdentifierAfterPackage);
    exit_on_parse_error();
    return;
  }

  bool library_parsed = false;
  if (auto library_token = ConsumeIf(TokenKind::Library)) {
    auto library_start = tree_->size();

    if (!ConsumeAndAddLeafNodeIf(TokenKind::StringLiteral,
                                 ParseNodeKind::Literal)) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      emitter_->Emit(*position_, ExpectedLibraryName);
      exit_on_parse_error();
      return;
    }

    AddNode(ParseNodeKind::PackageLibrary, *library_token, library_start,
            /*has_error=*/false);
    library_parsed = true;
  }

  switch (auto api_or_impl_token = tokens_->GetKind(*(position_))) {
    case TokenKind::Api: {
      AddLeafNode(ParseNodeKind::PackageApi, Consume());
      break;
    }
    case TokenKind::Impl: {
      AddLeafNode(ParseNodeKind::PackageImpl, Consume());
      break;
    }
    default: {
      if (!library_parsed && api_or_impl_token == TokenKind::StringLiteral) {
        // If we come acroess a string literal and we didn't parse `library
        // "..."` yet, then most probably the user forgot to add `library`
        // before the library name.
        CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                          "Missing `library` keyword.");
        emitter_->Emit(*position_, MissingLibraryKeyword);
      } else {
        CARBON_DIAGNOSTIC(ExpectedApiOrImpl, Error,
                          "Expected a `api` or `impl`.");
        emitter_->Emit(*position_, ExpectedApiOrImpl);
      }
      exit_on_parse_error();
      return;
    }
  }

  if (!PositionIs(TokenKind::Semi)) {
    CARBON_DIAGNOSTIC(ExpectedSemiToEndPackageDirective, Error,
                      "Expected `;` to end package directive.");
    emitter_->Emit(*position_, ExpectedSemiToEndPackageDirective);
    exit_on_parse_error();
    return;
  }

  AddNode(ParseNodeKind::PackageDirective, Consume(), state.subtree_start,
          /*has_error=*/false);
}

auto Parser::HandleParameter(ParserState pattern_state,
                             ParserState finish_state) -> void {
  PopAndDiscardState();

  PushState(finish_state);
  PushState(pattern_state);
}

auto Parser::HandleParameterAsDeducedState() -> void {
  HandleParameter(ParserState::PatternAsDeducedParameter,
                  ParserState::ParameterFinishAsDeduced);
}

auto Parser::HandleParameterAsRegularState() -> void {
  HandleParameter(ParserState::PatternAsParameter,
                  ParserState::ParameterFinishAsRegular);
}

auto Parser::HandleParameterFinish(TokenKind close_token,
                                   ParserState param_state) -> void {
  auto state = PopState();

  if (state.has_error) {
    ReturnErrorOnState();
  }

  if (ConsumeListToken(ParseNodeKind::ParameterListComma, close_token,
                       state.has_error) == ListTokenKind::Comma) {
    PushState(param_state);
  }
}

auto Parser::HandleParameterFinishAsDeducedState() -> void {
  HandleParameterFinish(TokenKind::CloseSquareBracket,
                        ParserState::ParameterAsDeduced);
}

auto Parser::HandleParameterFinishAsRegularState() -> void {
  HandleParameterFinish(TokenKind::CloseParen, ParserState::ParameterAsRegular);
}

auto Parser::HandleParameterList(ParseNodeKind parse_node_kind,
                                 TokenKind open_token_kind,
                                 TokenKind close_token_kind,
                                 ParserState param_state,
                                 ParserState finish_state) -> void {
  PopAndDiscardState();

  PushState(finish_state);
  AddLeafNode(parse_node_kind, ConsumeChecked(open_token_kind));

  if (!PositionIs(close_token_kind)) {
    PushState(param_state);
  }
}

auto Parser::HandleParameterListAsDeducedState() -> void {
  HandleParameterList(
      ParseNodeKind::DeducedParameterListStart, TokenKind::OpenSquareBracket,
      TokenKind::CloseSquareBracket, ParserState::ParameterAsDeduced,
      ParserState::ParameterListFinishAsDeduced);
}

auto Parser::HandleParameterListAsRegularState() -> void {
  HandleParameterList(ParseNodeKind::ParameterListStart, TokenKind::OpenParen,
                      TokenKind::CloseParen, ParserState::ParameterAsRegular,
                      ParserState::ParameterListFinishAsRegular);
}

auto Parser::HandleParameterListFinish(ParseNodeKind parse_node_kind,
                                       TokenKind token_kind) -> void {
  auto state = PopState();

  AddNode(parse_node_kind, ConsumeChecked(token_kind), state.subtree_start,
          state.has_error);
}

auto Parser::HandleParameterListFinishAsDeducedState() -> void {
  HandleParameterListFinish(ParseNodeKind::DeducedParameterList,
                            TokenKind::CloseSquareBracket);
}

auto Parser::HandleParameterListFinishAsRegularState() -> void {
  HandleParameterListFinish(ParseNodeKind::ParameterList,
                            TokenKind::CloseParen);
}

auto Parser::HandleParenCondition(ParseNodeKind start_kind,
                                  ParserState finish_state) -> void {
  auto state = PopState();

  ConsumeAndAddOpenParen(state.token, start_kind);

  state.state = finish_state;
  PushState(state);
  PushState(ParserState::Expression);
}

auto Parser::HandleParenConditionAsIfState() -> void {
  HandleParenCondition(ParseNodeKind::IfConditionStart,
                       ParserState::ParenConditionFinishAsIf);
}

auto Parser::HandleParenConditionAsWhileState() -> void {
  HandleParenCondition(ParseNodeKind::WhileConditionStart,
                       ParserState::ParenConditionFinishAsWhile);
}

auto Parser::HandleParenConditionFinishAsIfState() -> void {
  auto state = PopState();

  ConsumeAndAddCloseParen(state, ParseNodeKind::IfCondition);
}

auto Parser::HandleParenConditionFinishAsWhileState() -> void {
  auto state = PopState();

  ConsumeAndAddCloseParen(state, ParseNodeKind::WhileCondition);
}

auto Parser::HandleParenExpressionState() -> void {
  auto state = PopState();

  // Advance past the open paren.
  AddLeafNode(ParseNodeKind::ParenExpressionOrTupleLiteralStart,
              ConsumeChecked(TokenKind::OpenParen));

  if (PositionIs(TokenKind::CloseParen)) {
    state.state = ParserState::ParenExpressionFinishAsTuple;
    PushState(state);
  } else {
    state.state = ParserState::ParenExpressionFinishAsNormal;
    PushState(state);
    PushState(ParserState::ParenExpressionParameterFinishAsUnknown);
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleParenExpressionParameterFinish(bool as_tuple) -> void {
  auto state = PopState();

  auto list_token_kind = ConsumeListToken(
      ParseNodeKind::TupleLiteralComma, TokenKind::CloseParen, state.has_error);
  if (list_token_kind == ListTokenKind::Close) {
    return;
  }

  // If this is the first item and a comma was found, switch to tuple handling.
  // Note this could be `(expr,)` so we may not reuse the current state, but
  // it's still necessary to switch the parent.
  if (!as_tuple) {
    state.state = ParserState::ParenExpressionParameterFinishAsTuple;

    auto finish_state = PopState();
    CARBON_CHECK(finish_state.state ==
                 ParserState::ParenExpressionFinishAsNormal)
        << "Unexpected parent state, found: " << finish_state.state;
    finish_state.state = ParserState::ParenExpressionFinishAsTuple;
    PushState(finish_state);
  }

  // On a comma, push another expression handler.
  if (list_token_kind == ListTokenKind::Comma) {
    PushState(state);
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleParenExpressionParameterFinishAsUnknownState() -> void {
  HandleParenExpressionParameterFinish(/*as_tuple=*/false);
}

auto Parser::HandleParenExpressionParameterFinishAsTupleState() -> void {
  HandleParenExpressionParameterFinish(/*as_tuple=*/true);
}

auto Parser::HandleParenExpressionFinishAsNormalState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::ParenExpression, Consume(), state.subtree_start,
          state.has_error);
}

auto Parser::HandleParenExpressionFinishAsTupleState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::TupleLiteral, Consume(), state.subtree_start,
          state.has_error);
}

auto Parser::ConsumeIfPatternKeyword(TokenKind keyword_token,
                                     ParserState keyword_state,
                                     int subtree_start) -> void {
  if (auto token = ConsumeIf(keyword_token)) {
    PushState(StateStackEntry(
        keyword_state, PrecedenceGroup::ForTopLevelExpression(),
        PrecedenceGroup::ForTopLevelExpression(), *token, subtree_start));
  }
}

auto Parser::HandlePattern(PatternKind pattern_kind) -> void {
  auto state = PopState();

  // Parameters may have keywords prefixing the pattern. They become the parent
  // for the full PatternBinding.
  if (pattern_kind != PatternKind::Variable) {
    ConsumeIfPatternKeyword(TokenKind::Template, ParserState::PatternTemplate,
                            state.subtree_start);
    ConsumeIfPatternKeyword(TokenKind::Addr, ParserState::PatternAddress,
                            state.subtree_start);
  }

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&]() {
    switch (pattern_kind) {
      case PatternKind::DeducedParameter:
      case PatternKind::Parameter: {
        CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                          "Expected parameter declaration.");
        emitter_->Emit(*position_, ExpectedParameterName);
        break;
      }
      case PatternKind::Variable: {
        CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                          "Expected pattern in `var` declaration.");
        emitter_->Emit(*position_, ExpectedVariableName);
        break;
      }
    }
    // Add a placeholder for the type.
    AddLeafNode(ParseNodeKind::InvalidParse, *position_, /*has_error=*/true);
    state.state = ParserState::PatternFinishAsRegular;
    state.has_error = true;
    PushState(state);
  };

  // The first item should be an identifier or, for deduced parameters, `self`.
  bool has_name = false;
  if (auto identifier = ConsumeIf(TokenKind::Identifier)) {
    AddLeafNode(ParseNodeKind::DeclaredName, *identifier);
    has_name = true;
  } else if (pattern_kind == PatternKind::DeducedParameter) {
    if (auto self = ConsumeIf(TokenKind::SelfValueIdentifier)) {
      AddLeafNode(ParseNodeKind::SelfValueIdentifier, *self);
      has_name = true;
    }
  }
  if (!has_name) {
    // Add a placeholder for the name.
    AddLeafNode(ParseNodeKind::DeclaredName, *position_, /*has_error=*/true);
    on_error();
    return;
  }

  if (auto kind = PositionKind();
      kind == TokenKind::Colon || kind == TokenKind::ColonExclaim) {
    state.state = kind == TokenKind::Colon
                      ? ParserState::PatternFinishAsRegular
                      : ParserState::PatternFinishAsGeneric;
    // Use the `:` or `:!` for the root node.
    state.token = Consume();
    PushState(state);
    PushStateForExpression(PrecedenceGroup::ForType());
  } else {
    on_error();
    return;
  }
}

auto Parser::HandlePatternAsDeducedParameterState() -> void {
  HandlePattern(PatternKind::DeducedParameter);
}

auto Parser::HandlePatternAsParameterState() -> void {
  HandlePattern(PatternKind::Parameter);
}

auto Parser::HandlePatternAsVariableState() -> void {
  HandlePattern(PatternKind::Variable);
}

auto Parser::HandlePatternFinish(ParseNodeKind node_kind) -> void {
  auto state = PopState();

  AddNode(node_kind, state.token, state.subtree_start, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    ReturnErrorOnState();
  }
}

auto Parser::HandlePatternFinishAsGenericState() -> void {
  HandlePatternFinish(ParseNodeKind::GenericPatternBinding);
}

auto Parser::HandlePatternFinishAsRegularState() -> void {
  HandlePatternFinish(ParseNodeKind::PatternBinding);
}

auto Parser::HandlePatternAddressState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::Address, state.token, state.subtree_start,
          state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    ReturnErrorOnState();
  }
}

auto Parser::HandlePatternTemplateState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::Template, state.token, state.subtree_start,
          state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    ReturnErrorOnState();
  }
}

auto Parser::HandleStatementState() -> void {
  PopAndDiscardState();

  switch (PositionKind()) {
    case TokenKind::Break: {
      PushState(ParserState::StatementBreakFinish);
      AddLeafNode(ParseNodeKind::BreakStatementStart, Consume());
      break;
    }
    case TokenKind::Continue: {
      PushState(ParserState::StatementContinueFinish);
      AddLeafNode(ParseNodeKind::ContinueStatementStart, Consume());
      break;
    }
    case TokenKind::For: {
      PushState(ParserState::StatementForFinish);
      PushState(ParserState::StatementForHeader);
      ++position_;
      break;
    }
    case TokenKind::If: {
      PushState(ParserState::StatementIf);
      break;
    }
    case TokenKind::Return: {
      PushState(ParserState::StatementReturn);
      break;
    }
    case TokenKind::Var: {
      PushState(ParserState::VarAsSemicolon);
      break;
    }
    case TokenKind::While: {
      PushState(ParserState::StatementWhile);
      break;
    }
    default: {
      PushState(ParserState::ExpressionStatementFinish);
      PushState(ParserState::Expression);
      break;
    }
  }
}

auto Parser::HandleStatementBreakFinishState() -> void {
  HandleStatementKeywordFinish(ParseNodeKind::BreakStatement);
}

auto Parser::HandleStatementContinueFinishState() -> void {
  HandleStatementKeywordFinish(ParseNodeKind::ContinueStatement);
}

auto Parser::HandleStatementForHeaderState() -> void {
  auto state = PopState();

  ConsumeAndAddOpenParen(state.token, ParseNodeKind::ForHeaderStart);

  state.state = ParserState::StatementForHeaderIn;

  if (PositionIs(TokenKind::Var)) {
    PushState(state);
    PushState(ParserState::VarAsFor);
  } else {
    CARBON_DIAGNOSTIC(ExpectedVariableDeclaration, Error,
                      "Expected `var` declaration.");
    emitter_->Emit(*position_, ExpectedVariableDeclaration);

    if (auto next_in = FindNextOf({TokenKind::In})) {
      SkipTo(*next_in);
      ++position_;
    }
    state.has_error = true;
    PushState(state);
  }
}

auto Parser::HandleStatementForHeaderInState() -> void {
  auto state = PopState();

  state.state = ParserState::StatementForHeaderFinish;
  PushState(state);
  PushState(ParserState::Expression);
}

auto Parser::HandleStatementForHeaderFinishState() -> void {
  auto state = PopState();

  ConsumeAndAddCloseParen(state, ParseNodeKind::ForHeader);

  PushState(ParserState::CodeBlock);
}

auto Parser::HandleStatementForFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::ForStatement, state.token, state.subtree_start,
          state.has_error);
}

auto Parser::HandleStatementIfState() -> void {
  PopAndDiscardState();

  PushState(ParserState::StatementIfConditionFinish);
  PushState(ParserState::ParenConditionAsIf);
  ++position_;
}

auto Parser::HandleStatementIfConditionFinishState() -> void {
  auto state = PopState();

  state.state = ParserState::StatementIfThenBlockFinish;
  PushState(state);
  PushState(ParserState::CodeBlock);
}

auto Parser::HandleStatementIfThenBlockFinishState() -> void {
  auto state = PopState();

  if (ConsumeAndAddLeafNodeIf(TokenKind::Else,
                              ParseNodeKind::IfStatementElse)) {
    state.state = ParserState::StatementIfElseBlockFinish;
    PushState(state);
    // `else if` is permitted as a special case.
    PushState(PositionIs(TokenKind::If) ? ParserState::StatementIf
                                        : ParserState::CodeBlock);
  } else {
    AddNode(ParseNodeKind::IfStatement, state.token, state.subtree_start,
            state.has_error);
  }
}

auto Parser::HandleStatementIfElseBlockFinishState() -> void {
  auto state = PopState();
  AddNode(ParseNodeKind::IfStatement, state.token, state.subtree_start,
          state.has_error);
}

auto Parser::HandleStatementKeywordFinish(ParseNodeKind node_kind) -> void {
  auto state = PopState();

  auto semi = ConsumeIf(TokenKind::Semi);
  if (!semi) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    emitter_->Emit(*position_, ExpectedSemiAfter,
                   tokens_->GetKind(state.token));
    state.has_error = true;
    // Recover to the next semicolon if possible, otherwise indicate the
    // keyword for the error.
    semi = SkipPastLikelyEnd(state.token);
    if (!semi) {
      semi = state.token;
    }
  }
  AddNode(node_kind, *semi, state.subtree_start, state.has_error);
}

auto Parser::HandleStatementReturnState() -> void {
  auto state = PopState();
  state.state = ParserState::StatementReturnFinish;
  PushState(state);

  AddLeafNode(ParseNodeKind::ReturnStatementStart, Consume());
  if (!PositionIs(TokenKind::Semi)) {
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleStatementReturnFinishState() -> void {
  HandleStatementKeywordFinish(ParseNodeKind::ReturnStatement);
}

auto Parser::HandleStatementScopeLoopState() -> void {
  // This maintains the current state until we're at the end of the scope.

  auto token_kind = PositionKind();
  if (token_kind == TokenKind::CloseCurlyBrace) {
    auto state = PopState();
    if (state.has_error) {
      ReturnErrorOnState();
    }
  } else {
    PushState(ParserState::Statement);
  }
}

auto Parser::HandleStatementWhileState() -> void {
  PopAndDiscardState();

  PushState(ParserState::StatementWhileConditionFinish);
  PushState(ParserState::ParenConditionAsWhile);
  ++position_;
}

auto Parser::HandleStatementWhileConditionFinishState() -> void {
  auto state = PopState();

  state.state = ParserState::StatementWhileBlockFinish;
  PushState(state);
  PushState(ParserState::CodeBlock);
}

auto Parser::HandleStatementWhileBlockFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::WhileStatement, state.token, state.subtree_start,
          state.has_error);
}

auto Parser::HandleTypeIntroducer(ParseNodeKind introducer_kind,
                                  ParserState after_params_state) -> void {
  auto state = PopState();

  AddLeafNode(introducer_kind, Consume());

  state.state = after_params_state;
  PushState(state);
  state.state = ParserState::DeclarationNameAndParamsAsOptional;
  PushState(state);
}

auto Parser::HandleTypeIntroducerAsClassState() -> void {
  HandleTypeIntroducer(ParseNodeKind::ClassIntroducer,
                       ParserState::TypeAfterParamsAsClass);
}

auto Parser::HandleTypeIntroducerAsInterfaceState() -> void {
  HandleTypeIntroducer(ParseNodeKind::InterfaceIntroducer,
                       ParserState::TypeAfterParamsAsInterface);
}

auto Parser::HandleTypeIntroducerAsNamedConstraintState() -> void {
  HandleTypeIntroducer(ParseNodeKind::NamedConstraintIntroducer,
                       ParserState::TypeAfterParamsAsNamedConstraint);
}

auto Parser::HandleTypeAfterParams(ParseNodeKind declaration_kind,
                                   ParseNodeKind definition_start_kind,
                                   ParserState definition_finish_state)
    -> void {
  auto state = PopState();

  if (state.has_error) {
    HandleDeclarationError(state, declaration_kind,
                           /*skip_past_likely_end=*/true);
    return;
  }

  if (auto semi = ConsumeIf(TokenKind::Semi)) {
    AddNode(declaration_kind, *semi, state.subtree_start, state.has_error);
    return;
  }

  if (!PositionIs(TokenKind::OpenCurlyBrace)) {
    emitter_->Emit(*position_, ExpectedDeclarationSemiOrDefinition,
                   tokens_->GetKind(state.token));
    HandleDeclarationError(state, declaration_kind,
                           /*skip_past_likely_end=*/true);
    return;
  }

  state.state = definition_finish_state;
  PushState(state);
  PushState(ParserState::DeclarationScopeLoop);
  AddNode(definition_start_kind, Consume(), state.subtree_start,
          state.has_error);
}

auto Parser::HandleTypeAfterParamsAsClassState() -> void {
  HandleTypeAfterParams(ParseNodeKind::ClassDeclaration,
                        ParseNodeKind::ClassDefinitionStart,
                        ParserState::TypeDefinitionFinishAsClass);
}

auto Parser::HandleTypeAfterParamsAsInterfaceState() -> void {
  HandleTypeAfterParams(ParseNodeKind::InterfaceDeclaration,
                        ParseNodeKind::InterfaceDefinitionStart,
                        ParserState::TypeDefinitionFinishAsInterface);
}

auto Parser::HandleTypeAfterParamsAsNamedConstraintState() -> void {
  HandleTypeAfterParams(ParseNodeKind::NamedConstraintDeclaration,
                        ParseNodeKind::NamedConstraintDefinitionStart,
                        ParserState::TypeDefinitionFinishAsNamedConstraint);
}

auto Parser::HandleTypeDefinitionFinish(ParseNodeKind definition_kind) -> void {
  auto state = PopState();

  AddNode(definition_kind, Consume(), state.subtree_start, state.has_error);
}

auto Parser::HandleTypeDefinitionFinishAsClassState() -> void {
  HandleTypeDefinitionFinish(ParseNodeKind::ClassDefinition);
}

auto Parser::HandleTypeDefinitionFinishAsInterfaceState() -> void {
  HandleTypeDefinitionFinish(ParseNodeKind::InterfaceDefinition);
}

auto Parser::HandleTypeDefinitionFinishAsNamedConstraintState() -> void {
  HandleTypeDefinitionFinish(ParseNodeKind::NamedConstraintDefinition);
}

auto Parser::HandleVar(ParserState finish_state) -> void {
  PopAndDiscardState();

  // These will start at the `var`.
  PushState(finish_state);
  PushState(ParserState::VarAfterPattern);

  AddLeafNode(ParseNodeKind::VariableIntroducer, Consume());

  // This will start at the pattern.
  PushState(ParserState::PatternAsVariable);
}

auto Parser::HandleVarAsSemicolonState() -> void {
  HandleVar(ParserState::VarFinishAsSemicolon);
}

auto Parser::HandleVarAsForState() -> void {
  HandleVar(ParserState::VarFinishAsFor);
}

auto Parser::HandleVarAfterPatternState() -> void {
  auto state = PopState();

  if (state.has_error) {
    if (auto after_pattern = FindNextOf({TokenKind::Equal, TokenKind::Semi})) {
      SkipTo(*after_pattern);
    }
  }

  if (auto equals = ConsumeIf(TokenKind::Equal)) {
    AddLeafNode(ParseNodeKind::VariableInitializer, *equals);
    PushState(ParserState::Expression);
  }
}

auto Parser::HandleVarFinishAsSemicolonState() -> void {
  auto state = PopState();

  auto end_token = state.token;
  if (PositionIs(TokenKind::Semi)) {
    end_token = Consume();
  } else {
    emitter_->Emit(*position_, ExpectedSemiAfterExpression);
    state.has_error = true;
    if (auto semi_token = SkipPastLikelyEnd(state.token)) {
      end_token = *semi_token;
    }
  }
  AddNode(ParseNodeKind::VariableDeclaration, end_token, state.subtree_start,
          state.has_error);
}

auto Parser::HandleVarFinishAsForState() -> void {
  auto state = PopState();

  auto end_token = state.token;
  if (PositionIs(TokenKind::In)) {
    end_token = Consume();
  } else if (PositionIs(TokenKind::Colon)) {
    CARBON_DIAGNOSTIC(ExpectedInNotColon, Error,
                      "`:` should be replaced by `in`.");
    emitter_->Emit(*position_, ExpectedInNotColon);
    state.has_error = true;
    end_token = Consume();
  } else {
    CARBON_DIAGNOSTIC(ExpectedIn, Error,
                      "Expected `in` after loop `var` declaration.");
    emitter_->Emit(*position_, ExpectedIn);
    state.has_error = true;
  }

  AddNode(ParseNodeKind::ForIn, end_token, state.subtree_start,
          state.has_error);
}

}  // namespace Carbon

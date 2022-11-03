// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser2.h"

#include <cstdlib>
#include <memory>

#include "common/check.h"
#include "llvm/ADT/Optional.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

Parser2::Parser2(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                 TokenDiagnosticEmitter& emitter)
    : tree_(tree_arg),
      tokens_(tokens_arg),
      emitter_(emitter),
      position_(tokens_.tokens().begin()),
      end_(tokens_.tokens().end()) {
  CARBON_CHECK(position_ != end_) << "Empty TokenizedBuffer";
  --end_;
  CARBON_CHECK(tokens_.GetKind(*end_) == TokenKind::EndOfFile())
      << "TokenizedBuffer should end with EndOfFile, ended with "
      << tokens_.GetKind(*end_).Name();
}

auto Parser2::AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                          bool has_error) -> void {
  tree_.node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, /*subtree_size=*/1));
  if (has_error) {
    tree_.has_errors_ = true;
  }
}

auto Parser2::AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                      int subtree_start, bool has_error) -> void {
  int subtree_size = tree_.size() - subtree_start + 1;
  tree_.node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, subtree_size));
  if (has_error) {
    tree_.has_errors_ = true;
  }
}

auto Parser2::ConsumeAndAddLeafNodeIf(TokenKind token_kind,
                                      ParseNodeKind node_kind) -> bool {
  auto token = ConsumeIf(token_kind);
  if (!token) {
    return false;
  }

  AddLeafNode(node_kind, *token);
  return true;
}

auto Parser2::ConsumeIf(TokenKind kind)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (!PositionIs(kind)) {
    return llvm::None;
  }
  auto token = *position_;
  ++position_;
  return token;
}

auto Parser2::FindNextOf(std::initializer_list<TokenKind> desired_kinds)
    -> llvm::Optional<TokenizedBuffer::Token> {
  auto new_position = position_;
  while (true) {
    TokenizedBuffer::Token token = *new_position;
    TokenKind kind = tokens_.GetKind(token);
    if (kind.IsOneOf(desired_kinds)) {
      return token;
    }

    // Step to the next token at the current bracketing level.
    if (kind.IsClosingSymbol() || kind == TokenKind::EndOfFile()) {
      // There are no more tokens at this level.
      return llvm::None;
    } else if (kind.IsOpeningSymbol()) {
      new_position =
          TokenizedBuffer::TokenIterator(tokens_.GetMatchedClosingToken(token));
      // Advance past the closing token.
      ++new_position;
    } else {
      ++new_position;
    }
  }
}

auto Parser2::Parse() -> void {
  PushState(ParserState::Declaration());
  while (!state_stack_.empty()) {
    switch (state_stack_.back().state) {
#define CARBON_PARSER_STATE(Name) \
  case ParserState::Name():       \
    Handle##Name##State();        \
    break;
#include "toolchain/parser/parser_state.def"
    }
  }

  AddLeafNode(ParseNodeKind::FileEnd(), *position_);
}

auto Parser2::SkipMatchingGroup() -> bool {
  if (!PositionKind().IsOpeningSymbol()) {
    return false;
  }

  SkipTo(tokens_.GetMatchedClosingToken(*position_));
  ++position_;
  return true;
}

auto Parser2::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (position_ == end_) {
    return llvm::None;
  }

  TokenizedBuffer::Line root_line = tokens_.GetLine(skip_root);
  int root_line_indent = tokens_.GetIndentColumnNumber(root_line);

  // We will keep scanning through tokens on the same line as the root or
  // lines with greater indentation than root's line.
  auto is_same_line_or_indent_greater_than_root =
      [&](TokenizedBuffer::Token t) {
        TokenizedBuffer::Line l = tokens_.GetLine(t);
        if (l == root_line) {
          return true;
        }

        return tokens_.GetIndentColumnNumber(l) > root_line_indent;
      };

  do {
    if (PositionIs(TokenKind::CloseCurlyBrace())) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return llvm::None;
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(TokenKind::Semi())) {
      return semi;
    }

    // Skip over any matching group of tokens_.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position_;
  } while (position_ != end_ &&
           is_same_line_or_indent_greater_than_root(*position_));

  return llvm::None;
}

auto Parser2::SkipTo(TokenizedBuffer::Token t) -> void {
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards from " << position_
                                << " to " << TokenizedBuffer::TokenIterator(t);
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

auto Parser2::HandleDeclarationState() -> void {
  // This maintains the current state unless we're at the end of the file.

  switch (PositionKind()) {
    case TokenKind::EndOfFile(): {
      state_stack_.pop_back();
      break;
    }
    case TokenKind::Fn(): {
      PushState(ParserState::FunctionIntroducer());
      AddLeafNode(ParseNodeKind::FunctionIntroducer(), *position_);
      ++position_;
      break;
    }
    case TokenKind::Semi(): {
      AddLeafNode(ParseNodeKind::EmptyDeclaration(), *position_);
      ++position_;
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                        "Unrecognized declaration introducer.");
      emitter_.Emit(*position_, UnrecognizedDeclaration);
      tree_.has_errors_ = true;
      if (auto semi = SkipPastLikelyEnd(*position_)) {
        AddLeafNode(ParseNodeKind::EmptyDeclaration(), *semi,
                    /*has_error=*/true);
      }
      break;
    }
  }
}

auto Parser2::HandleExpressionPrimary() -> void {
  // TODO: Handle OpenParen and OpenCurlyBrace.
  switch (PositionKind()) {
    case TokenKind::Identifier():
      AddLeafNode(ParseNodeKind::NameReference(), *position_);
      break;

    case TokenKind::IntegerLiteral():
    case TokenKind::RealLiteral():
    case TokenKind::StringLiteral():
    case TokenKind::IntegerTypeLiteral():
    case TokenKind::UnsignedIntegerTypeLiteral():
    case TokenKind::FloatingPointTypeLiteral():
      AddLeafNode(ParseNodeKind::Literal(), *position_);
      break;

    default:
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      emitter_.Emit(*position_, ExpectedExpression);
      break;
  }
  ++position_;
}

auto Parser2::HandleExpressionState() -> void {
  // TODO: This is temporary, we should need this state. If not, maybe add an
  // overload that uses pop_back instead of pop_back_val.
  auto state = PopState();
  (void)state;

  HandleExpressionPrimary();
}

auto Parser2::HandleExpressionForTypeState() -> void {
  // TODO: This is temporary, we should need this state. If not, maybe add an
  // overload that uses pop_back instead of pop_back_val.
  auto state = PopState();
  (void)state;

  HandleExpressionPrimary();
}

auto Parser2::HandleFunctionError(StateStackEntry state,
                                  bool skip_past_likely_end) -> void {
  auto token = state.token;
  if (skip_past_likely_end) {
    if (auto semi = SkipPastLikelyEnd(token)) {
      token = *semi;
    }
  }
  AddNode(ParseNodeKind::FunctionDeclaration(), token, state.subtree_start,
          /*has_error=*/true);
}

auto Parser2::HandleFunctionIntroducerState() -> void {
  auto state = PopState();

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                               ParseNodeKind::DeclaredName())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionName, Error,
                      "Expected function name after `fn` keyword.");
    emitter_.Emit(*position_, ExpectedFunctionName);
    // TODO: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    HandleFunctionError(state, true);
    return;
  }

  if (!PositionIs(TokenKind::OpenParen())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionParams, Error,
                      "Expected `(` after function name.");
    emitter_.Emit(*position_, ExpectedFunctionParams);
    HandleFunctionError(state, true);
    return;
  }

  // Parse the parameter list as its own subtree; once that pops, resume
  // function parsing.
  state.state = ParserState::FunctionAfterParameterList();
  PushState(state);
  // TODO: When swapping () start/end, this should AddLeafNode the open before
  // continuing.
  PushState(ParserState::FunctionParameterListFinish());
  // Advance past the open paren.
  ++position_;
  if (PositionKind() != TokenKind::CloseParen()) {
    PushState(ParserState::PatternForFunctionParameter());
  }
}

auto Parser2::HandleFunctionParameterListFinishState() -> void {
  auto state = PopState();

  CARBON_CHECK(PositionKind() == TokenKind::CloseParen())
      << PositionKind().Name();
  AddLeafNode(ParseNodeKind::ParameterListEnd(), *position_);
  AddNode(ParseNodeKind::ParameterList(), state.token, state.subtree_start,
          state.has_error);
  ++position_;
}

auto Parser2::HandleFunctionAfterParameterListState() -> void {
  auto state = PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  state.state = ParserState::FunctionSignatureFinish();
  PushState(state);

  // If there is a return type, parse the expression before adding the return
  // type nod.e
  if (PositionIs(TokenKind::MinusGreater())) {
    PushState(ParserState::FunctionReturnTypeFinish());
    ++position_;
    PushState(ParserState::ExpressionForType());
  }
}

auto Parser2::HandleFunctionReturnTypeFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::ReturnType(), state.token, state.subtree_start);
}

auto Parser2::HandleFunctionSignatureFinishState() -> void {
  auto state = PopState();

  switch (PositionKind()) {
    case TokenKind::Semi(): {
      AddNode(ParseNodeKind::FunctionDeclaration(), *position_,
              state.subtree_start);
      ++position_;
      break;
    }
    case TokenKind::OpenCurlyBrace(): {
      AddNode(ParseNodeKind::FunctionDefinitionStart(), *position_,
              state.subtree_start);
      state.state = ParserState::FunctionDefinitionFinish();
      PushState(state);
      PushState(ParserState::StatementScope());
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(
          ExpectedFunctionBodyOrSemi, Error,
          "Expected function definition or `;` after function declaration.");
      emitter_.Emit(*position_, ExpectedFunctionBodyOrSemi);
      // Only need to skip if we've not already found a new line.
      bool skip_past_likely_end =
          tokens_.GetLine(*position_) == tokens_.GetLine(state.token);
      HandleFunctionError(state, skip_past_likely_end);
      break;
    }
  }
}

auto Parser2::HandleFunctionDefinitionFinishState() -> void {
  auto state = PopState();
  AddNode(ParseNodeKind::FunctionDefinition(), *position_, state.subtree_start);
  ++position_;
}

auto Parser2::HandlePatternStart(PatternKind pattern_kind) -> void {
  auto state = PopState();

  // Ensure the finish state always follows.
  switch (pattern_kind) {
    case PatternKind::Parameter: {
      state.state = ParserState::PatternForFunctionParameterFinish();
      break;
    }
    case PatternKind::Variable: {
      CARBON_FATAL() << "TODO";
      break;
    }
  }

  // Handle an invalid pattern introducer.
  if (!PositionIs(TokenKind::Identifier()) ||
      tokens_.GetKind(*(position_ + 1)) != TokenKind::Colon()) {
    switch (pattern_kind) {
      case PatternKind::Parameter: {
        CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                          "Expected parameter declaration.");
        emitter_.Emit(*position_, ExpectedParameterName);
        break;
      }
      case PatternKind::Variable: {
        CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                          "Expected pattern in `var` declaration.");
        emitter_.Emit(*position_, ExpectedVariableName);
        break;
      }
    }
    state.has_error = true;
    PushState(state);
    return;
  }

  // Switch the context token to the colon, so that it'll be used for the root
  // node.
  state.token = *(position_ + 1);
  PushState(state);
  PushState(ParserState::ExpressionForType());
  AddLeafNode(ParseNodeKind::DeclaredName(), *position_);
  position_ += 2;
}

auto Parser2::HandlePatternForFunctionParameterState() -> void {
  HandlePatternStart(PatternKind::Parameter);
}

auto Parser2::HandlePatternForFunctionParameterFinishState() -> void {
  auto state = PopState();

  // If an error was encountered, propagate it without adding a node.
  bool has_error = state.has_error;
  if (has_error) {
    ReturnErrorOnState();
  } else {
    // TODO: may need to mark has_error if !type.
    AddNode(ParseNodeKind::PatternBinding(), state.token, state.subtree_start);
  }

  // Handle tokens following a parameter.
  switch (PositionKind()) {
    case TokenKind::CloseParen(): {
      // Done with the parameter list.
      return;
    }
    case TokenKind::Comma(): {
      // Comma handling is after the switch.
      break;
    }
    default: {
      // Don't error twice for the same issue.
      if (!has_error) {
        CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                          "Expected `,` or `)`.");
        emitter_.Emit(*position_, UnexpectedTokenAfterListElement);
        ReturnErrorOnState();
      }

      // Recover from the invalid token.
      auto end_of_element =
          FindNextOf({TokenKind::Comma(), TokenKind::CloseParen()});
      // The lexer guarantees that parentheses are balanced.
      CARBON_CHECK(end_of_element) << "missing matching `)` for `(`";
      SkipTo(*end_of_element);
      if (PositionKind() == TokenKind::CloseParen()) {
        // Done with the parameter list.
        return;
      }
      // Comma handling is after the switch.
      break;
    }
  }

  // We are guaranteed to now be at a comma.
  AddLeafNode(ParseNodeKind::ParameterListComma(), *position_);
  ++position_;
  if (PositionKind() != TokenKind::CloseParen()) {
    PushState(ParserState::PatternForFunctionParameter());
  }
}

auto Parser2::HandleReturnStatementState() -> void {
  auto state = PopState();

  auto semi =
      ConsumeAndAddLeafNodeIf(TokenKind::Semi(), ParseNodeKind::StatementEnd());
  if (!semi) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedSemiAfter, TokenKind::Return());
    // TODO: Try to skip to a semicolon to recover.
  }
  AddNode(ParseNodeKind::ReturnStatement(), state.token, state.subtree_start);
  ++position_;
}

auto Parser2::HandleStatementScopeState() -> void {
  switch (PositionKind()) {
    case TokenKind::If(): {
      return;
    }
    case TokenKind::Return(): {
      auto start = *position_;
      if (tokens_.GetKind(*(position_ + 1)) == TokenKind::Semi()) {
        int subtree_start = tree_.size();
        AddLeafNode(ParseNodeKind::StatementEnd(), *(position_ + 1));
        AddNode(ParseNodeKind::ReturnStatement(), start, subtree_start);
      } else {
        PushState(ParserState::ReturnStatement());
        PushState(ParserState::Expression());
      }
      return;
    }
    default: {
      CARBON_FATAL() << "TODO: Parse as expression";
    }
  }
}

}  // namespace Carbon

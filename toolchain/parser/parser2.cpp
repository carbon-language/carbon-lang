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
  state_stack_.pop_back();
}

auto Parser2::HandleExpressionState() -> void {
  // TODO: This is temporary.
  HandleExpressionPrimary();
}

auto Parser2::HandleExpressionForTypeState() -> void {
  // TODO: This is temporary.
  HandleExpressionPrimary();
}

auto Parser2::HandleFunctionError(bool skip_past_likely_end) -> void {
  auto token = state_stack_.back().token;
  if (skip_past_likely_end) {
    if (auto semi = SkipPastLikelyEnd(token)) {
      token = *semi;
    }
  }
  AddNode(ParseNodeKind::FunctionDeclaration(), token,
          state_stack_.back().subtree_start,
          /*has_error=*/true);
  state_stack_.pop_back();
}

auto Parser2::HandleFunctionIntroducerState() -> void {
  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                               ParseNodeKind::DeclaredName())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionName, Error,
                      "Expected function name after `fn` keyword.");
    emitter_.Emit(*position_, ExpectedFunctionName);
    // TODO: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    HandleFunctionError(true);
    return;
  }

  if (!PositionIs(TokenKind::OpenParen())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionParams, Error,
                      "Expected `(` after function name.");
    emitter_.Emit(*position_, ExpectedFunctionParams);
    HandleFunctionError(true);
    return;
  }

  // Parse the parameter list as its own subtree; once that pops, resume
  // function parsing.
  state_stack_.back().state = ParserState::FunctionParameterListFinish();
  PushState(ParserState::FunctionParameterListStart());
  // Advance past the open parenthesis before continuing.
  // TODO: When swapping () start/end, this should AddNode the open before
  // continuing.
  ++position_;
}

auto Parser2::HandleFunctionParameterList(bool is_start) -> void {
  auto token_kind = PositionKind();

  if (!is_start) {
    // Handle tokens following a parameter.
    if (token_kind == TokenKind::Comma()) {
      AddLeafNode(ParseNodeKind::ParameterListComma(), *position_);
      ++position_;
      token_kind = PositionKind();
    } else if (token_kind != TokenKind::CloseParen()) {
      CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                        "Expected `,` or `)`.");
      emitter_.Emit(*position_, UnexpectedTokenAfterListElement);
      // TODO: Finish
      return;
    }
  }

  if (token_kind == TokenKind::CloseParen()) {
    AddLeafNode(ParseNodeKind::ParameterListEnd(), *position_);
    AddNode(ParseNodeKind::ParameterList(), state_stack_.back().token,
            state_stack_.back().subtree_start);
    ++position_;
    state_stack_.pop_back();
    return;
  }

  state_stack_.back().state = ParserState::FunctionParameterListResume();
  PushState(ParserState::PatternForFunctionParameter());
}

auto Parser2::HandleFunctionParameterListStartState() -> void {
  HandleFunctionParameterList(true);
}

auto Parser2::HandleFunctionParameterListResumeState() -> void {
  HandleFunctionParameterList(false);
}

auto Parser2::HandleFunctionParameterListFinishState() -> void {
  // Regardless of whether there's a return type, we'll finish the signature.
  state_stack_.back().state = ParserState::FunctionSignatureFinish();
  // If there is a return type, parse the expression before adding the return
  // type nod.e
  if (PositionIs(TokenKind::MinusGreater())) {
    PushState(ParserState::FunctionReturnTypeFinish());
    ++position_;
    PushState(ParserState::ExpressionForType());
  }
}

auto Parser2::HandleFunctionReturnTypeFinishState() -> void {
  AddNode(ParseNodeKind::ReturnType(), state_stack_.back().token,
          state_stack_.back().subtree_start);
  state_stack_.pop_back();
}

auto Parser2::HandleFunctionSignatureFinishState() -> void {
  switch (PositionKind()) {
    case TokenKind::Semi(): {
      AddNode(ParseNodeKind::FunctionDeclaration(), *position_,
              state_stack_.back().subtree_start);
      ++position_;
      state_stack_.pop_back();
      break;
    }
    case TokenKind::OpenCurlyBrace(): {
      AddNode(ParseNodeKind::FunctionDefinitionStart(), *position_,
              state_stack_.back().subtree_start);
      state_stack_.back().state = ParserState::FunctionDefinitionFinish();
      PushState(ParserState::StatementScope());
      break;
    }
    default: {
      llvm::errs() << PositionKind().Name() << "\n";
      CARBON_DIAGNOSTIC(
          ExpectedFunctionBodyOrSemi, Error,
          "Expected function definition or `;` after function declaration.");
      emitter_.Emit(*position_, ExpectedFunctionBodyOrSemi);
      // Only need to skip if we've not already found a new line.
      HandleFunctionError(tokens_.GetLine(*position_) ==
                          tokens_.GetLine(state_stack_.back().token));
      break;
    }
  }
}

auto Parser2::HandleFunctionDefinitionFinishState() -> void {
  AddNode(ParseNodeKind::FunctionDefinition(), *position_,
          state_stack_.back().subtree_start);
  state_stack_.pop_back();
  ++position_;
}

auto Parser2::HandlePatternStart(PatternKind pattern_kind) -> void {
  // Ensure the finish state always follows.
  switch (pattern_kind) {
    case PatternKind::Parameter: {
      state_stack_.back().state =
          ParserState::PatternForFunctionParameterFinish();
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
    state_stack_.back().has_error = true;
    return;
  }

  // Switch the context token to the colon, so that it'll be used for the root
  // node.
  state_stack_.back().token = *(position_ + 1);
  PushState(ParserState::ExpressionForType());
  AddLeafNode(ParseNodeKind::DeclaredName(), *position_);
  position_ += 2;
}

auto Parser2::HandlePatternForFunctionParameterState() -> void {
  HandlePatternStart(PatternKind::Parameter);
}

auto Parser2::HandlePatternForFunctionParameterFinishState() -> void {
  // If an error was encountered, propagate it without adding a node.
  if (state_stack_.back().has_error) {
    state_stack_.pop_back();
    state_stack_.back().has_error = true;
    return;
  }
  // TODO: may need to mark has_error if !type.
  AddNode(ParseNodeKind::PatternBinding(), state_stack_.back().token,
          state_stack_.back().subtree_start);
  state_stack_.pop_back();
}

auto Parser2::HandleReturnStatementState() -> void {
  auto semi =
      ConsumeAndAddLeafNodeIf(TokenKind::Semi(), ParseNodeKind::StatementEnd());
  if (!semi) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedSemiAfter, TokenKind::Return());
    // TODO: Try to skip to a semicolon to recover.
  }
  AddNode(ParseNodeKind::ReturnStatement(), state_stack_.back().token,
          state_stack_.back().subtree_start);
  state_stack_.pop_back();
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

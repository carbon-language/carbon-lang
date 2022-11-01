// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser2.h"

#include <cstdlib>
#include <memory>

#include "common/check.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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
      << tokens_.GetKind(*end_);
  (void)tree_;
  (void)emitter_;
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
  if (!NextTokenIs(kind)) {
    return {};
  }
  auto token = *position_;
  ++position_;
  return token;
}

auto Parser2::Parse() -> void {
  PushState(ParserState::Declaration());
  while (position_ < end_) {
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

auto Parser2::SkipMatchingGroup(TokenizedBuffer::Token token) -> bool {
  if (!tokens_.GetKind(token).IsOpeningSymbol()) {
    return false;
  }

  SkipTo(tokens_.GetMatchedClosingToken(token));
  return true;
}

auto Parser2::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root) -> bool {
  CARBON_CHECK(position_ != end_);

  TokenizedBuffer::Line root_line = tokens_.GetLine(skip_root);
  int root_line_indent = tokens_.GetIndentColumnNumber(root_line);

  while (true) {
    auto next = position_ + 1;
    if (next == end_) {
      return false;
    }

    switch (tokens_.GetKind(*next)) {
      case TokenKind::CloseCurlyBrace(): {
        // Immediately bail out if we hit an unmatched close curly, this will
        // pop us up a level of the syntax grouping.
        return false;
      }
      case TokenKind::Semi(): {
        // Advance to the semi.
        ++position_;
        return true;
      }
      default: {
        // We will keep scanning through tokens on the same line as the root or
        // lines with greater indentation than root's line.
        TokenizedBuffer::Line l = tokens_.GetLine(*next);
        if (l != root_line ||
            tokens_.GetIndentColumnNumber(l) > root_line_indent) {
          return false;
        }

        // Skip over any matching group of tokens_.
        SkipMatchingGroup(*next);
        ++position_;
      }
    }
  }
}

auto Parser2::SkipTo(TokenizedBuffer::Token t) -> void {
  CARBON_CHECK(t > *position_) << "Tried to skip backwards.";
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

auto Parser2::HandleDeclarationState() -> void {
  do {
    switch (auto token_kind = tokens_.GetKind(*position_)) {
      case TokenKind::Fn(): {
        AddLeafNode(ParseNodeKind::FunctionIntroducer(), *position_);
        ++position_;
        PushState(ParserState::FunctionIntroducer());
        return;
      }
      case TokenKind::Semi(): {
        AddLeafNode(ParseNodeKind::EmptyDeclaration(), *position_);
        break;
      }
      default: {
        CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                          "Unrecognized declaration introducer.");
        emitter_.Emit(*position_, UnrecognizedDeclaration);
        tree_.has_errors_ = true;
        if (SkipPastLikelyEnd(*position_)) {
          AddLeafNode(ParseNodeKind::EmptyDeclaration(), *position_,
                      /*has_error=*/true);
        }
        break;
      }
    }
    ++position_;
  } while (position_ < end_);
}

auto Parser2::HandleFunctionIntroducerState() -> void {
  // When handling errors before the start of the definition, treat it as a
  // declaration. Recover to a semicolon when it makes sense as a possible
  // function end, otherwise use the fn token for the error.
  auto add_error_function_node = [&](bool skip_past_likely_end) {
    auto function_intro_token = state_stack_.back().start_token;
    auto subtree_start = state_stack_.back().subtree_start;
    if (skip_past_likely_end && SkipPastLikelyEnd(function_intro_token)) {
      AddNode(ParseNodeKind::FunctionDeclaration(), *position_, subtree_start,
              /*has_error=*/true);
      ++position_;
      return;
    }
    AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
            subtree_start, /*has_error=*/true);
  };

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                               ParseNodeKind::DeclaredName())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionName, Error,
                      "Expected function name after `fn` keyword.");
    emitter_.Emit(*position_, ExpectedFunctionName);
    // TODO: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    return add_error_function_node(true);
  }

  TokenizedBuffer::Token open_paren = *position_;
  if (tokens_.GetKind(open_paren) != TokenKind::OpenParen()) {
    CARBON_DIAGNOSTIC(ExpectedFunctionParams, Error,
                      "Expected `(` after function name.");
    emitter_.Emit(open_paren, ExpectedFunctionParams);
    add_error_function_node(true);
    return;
  }
  /*
  TokenizedBuffer::Token close_paren =
      tokens_.GetMatchedClosingToken(open_paren);
  */
  add_error_function_node(true);
}

}  // namespace Carbon

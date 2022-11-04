// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER2_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER2_H_

#include "llvm/ADT/Optional.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/parser/parser_state.h"

namespace Carbon {

class Parser2 {
 public:
  // Parses the tokens into a parse tree, emitting any errors encountered.
  //
  // This is the entry point to the parser implementation.
  static auto Parse(TokenizedBuffer& tokens, TokenDiagnosticEmitter& emitter)
      -> ParseTree {
    ParseTree tree(tokens);
    Parser2 parser(tree, tokens, emitter);
    parser.Parse();
    return tree;
  }

 private:
  class PrettyStackTraceParseState;

  // Used to track state on state_stack_.
  struct StateStackEntry {
    // The state.
    ParserState state;
    // The token indicating the start of a tracked subtree.
    TokenizedBuffer::Token start_token;
    // The offset within the ParseTree of the subtree start.
    int32_t subtree_start;
  };

  Parser2(ParseTree& tree, TokenizedBuffer& tokens,
          TokenDiagnosticEmitter& emitter);

  auto Parse() -> void;

  // Adds a node to the parse tree that is fully parsed, has no children
  // ("leaf"), and has a subsequent sibling.
  //
  // This sets up the next sibling of the node to be the next node in the parse
  // tree's preorder sequence.
  auto AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                   bool has_error = false) -> void;

  auto AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
               int subtree_start, bool has_error = false) -> void;

  // Composes `ConsumeIf` and `AddLeafNode`, returning false when ConsumeIf
  // fails.
  auto ConsumeAndAddLeafNodeIf(TokenKind token_kind, ParseNodeKind node_kind)
      -> bool;

  // If the current position's token matches this `Kind`, returns it and
  // advances to the next position. Otherwise returns an empty optional.
  auto ConsumeIf(TokenKind kind) -> llvm::Optional<TokenizedBuffer::Token>;

  // Gets the kind of the next token to be consumed.
  auto PositionKind() const -> TokenKind { return tokens_.GetKind(*position_); }

  // Tests whether the next token to be consumed is of the specified kind.
  auto PositionIs(TokenKind kind) const -> bool {
    return PositionKind() == kind;
  }

  // If the token is an opening symbol for a matched group, skips to the matched
  // closing symbol and returns true. Otherwise, returns false.
  auto SkipMatchingGroup() -> bool;

  // Skips forward to move past the likely end of a declaration or statement.
  //
  // Looks forward, skipping over any matched symbol groups, to find the next
  // position that is likely past the end of a declaration or statement. This
  // is a heuristic and should only be called when skipping past parse errors.
  //
  // The strategy for recognizing when we have likely passed the end of a
  // declaration or statement:
  // - If we get to a close curly brace, we likely ended the entire context.
  // - If we get to a semicolon, that should have ended the declaration or
  //   statement.
  // - If we get to a new line from the `SkipRoot` token, but with the same or
  //   less indentation, there is likely a missing semicolon. Continued
  //   declarations or statements across multiple lines should be indented.
  //
  // Returns a semicolon token if one is the likely end.
  auto SkipPastLikelyEnd(TokenizedBuffer::Token skip_root)
      -> llvm::Optional<TokenizedBuffer::Token>;

  // Skip forward to the given token. Verifies that it is actually forward.
  auto SkipTo(TokenizedBuffer::Token t) -> void;

  auto PushState(ParserState state) -> void {
    state_stack_.push_back({state, *position_, tree_.size()});
  }

  // When handling errors before the start of the definition, treat it as a
  // declaration. Recover to a semicolon when it makes sense as a possible
  // function end, otherwise use the fn token for the error.
  auto HandleFunctionError(bool skip_past_likely_end) -> void;

#define CARBON_PARSER_STATE(Name) auto Handle##Name##State()->void;
#include "toolchain/parser/parser_state.def"

  ParseTree& tree_;
  TokenizedBuffer& tokens_;
  TokenDiagnosticEmitter& emitter_;

  // The current position within the token buffer.
  TokenizedBuffer::TokenIterator position_;
  // The EndOfFile token.
  TokenizedBuffer::TokenIterator end_;

  llvm::SmallVector<StateStackEntry> state_stack_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER2_H_

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
  // Supported kinds of patterns for HandlePattern.
  enum class PatternKind { Parameter, Variable };

  // Helper class for tracing state_stack_ on crashes.
  class PrettyStackTraceParseState;

  // Used to track state on state_stack_.
  struct StateStackEntry {
    StateStackEntry(ParserState state, TokenizedBuffer::Token token,
                    int32_t subtree_start)
        : state(state), token(token), subtree_start(subtree_start) {}

    // The state.
    ParserState state;
    // A token providing context based on the subtree. This will typically be
    // the first token in the subtree, but may sometimes be a token within. It
    // will typically be used for the subtree's root node.
    TokenizedBuffer::Token token;
    // The offset within the ParseTree of the subtree start.
    int32_t subtree_start;
    // Set to true  to indicate that an error was found, and that contextual
    // error recovery may be needed.
    bool has_error = false;
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
               int subtree_start, bool has_error) -> void;

  // Parses a close paren token corresponding to the given open paren token,
  // possibly skipping forward and diagnosing if necessary. Creates a parse node
  // of the specified kind if successful.
  auto ConsumeAndAddCloseParen(TokenizedBuffer::Token open_paren,
                               ParseNodeKind close_kind) -> bool;

  // Composes `ConsumeIf` and `AddLeafNode`, returning false when ConsumeIf
  // fails.
  auto ConsumeAndAddLeafNodeIf(TokenKind token_kind, ParseNodeKind node_kind)
      -> bool;

  // If the current position's token matches this `Kind`, returns it and
  // advances to the next position. Otherwise returns an empty optional.
  auto ConsumeIf(TokenKind kind) -> llvm::Optional<TokenizedBuffer::Token>;

  // Find the next token of any of the given kinds at the current bracketing
  // level.
  auto FindNextOf(std::initializer_list<TokenKind> desired_kinds)
      -> llvm::Optional<TokenizedBuffer::Token>;

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

  // Pushes a new state with the current position for context.
  auto PushState(ParserState state) -> void {
    PushState(StateStackEntry(state, *position_, tree_.size()));
  }

  // Pushes a new state with the token for context.
  auto PushState(ParserState state, TokenizedBuffer::Token token) -> void {
    PushState(StateStackEntry(state, token, tree_.size()));
  }

  // Pushes a constructed state onto the stack.
  auto PushState(StateStackEntry state) -> void {
    state_stack_.push_back(state);
  }

  // Pops the state and keeps the value for inspection.
  auto PopState() -> StateStackEntry { return state_stack_.pop_back_val(); }

  // Pops the state and discards it.
  auto PopAndDiscardState() -> void { state_stack_.pop_back(); }

  // Propagates an error up the state stack, to the parent state.
  auto ReturnErrorOnState() -> void { state_stack_.back().has_error = true; }

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  auto HandleExpressionFormPrimary() -> void;

  // When handling errors before the start of the definition, treat it as a
  // declaration. Recover to a semicolon when it makes sense as a possible
  // function end, otherwise use the fn token for the error.
  auto HandleFunctionError(StateStackEntry state, bool skip_past_likely_end)
      -> void;

  // Handles a code block in the context of a statement scope.
  auto HandleCodeBlock() -> void;

  // Handles parsing of a function parameter list, including commas and the
  // close paren.
  auto HandleFunctionParameterList(bool is_start) -> void;

  // Handles the `;` after a keyword statement.
  auto HandleKeywordStatementFinish(TokenKind token_kind,
                                    ParseNodeKind node_kind) -> void;

  // Handles the start of a pattern.
  // If the start of the pattern is invalid, it's the responsibility of the
  // outside context to advance past the pattern.
  auto HandlePatternStart(PatternKind pattern_kind) -> void;

  // Handles a single statement. While typically within a statement block, this
  // can also be used for error recovery where we expect a statement block and
  // are missing braces.
  auto HandleStatement(TokenKind token_kind) -> void;

  // Handles a `if` statement at the start `if` token.
  auto HandleStatementIf() -> void;

  // `clang-format` has a bug with spacing around `->` returns in macros. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
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

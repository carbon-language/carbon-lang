// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_H_

#include <optional>

#include "common/check.h"
#include "common/vlog.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/parser/parser_state.h"
#include "toolchain/parser/precedence.h"

namespace Carbon {

// This parser uses a stack for state transitions. See parser_state.def for
// state documentation.
class Parser {
 public:
  // Parses the tokens into a parse tree, emitting any errors encountered.
  //
  // This is the entry point to the parser implementation.
  static auto Parse(TokenizedBuffer& tokens, TokenDiagnosticEmitter& emitter,
                    llvm::raw_ostream* vlog_stream) -> ParseTree {
    ParseTree tree(tokens);
    Parser parser(tree, tokens, emitter, vlog_stream);
    parser.Parse();
    return tree;
  }

 private:
  // Possible operator fixities for errors.
  enum class OperatorFixity { Prefix, Infix, Postfix };

  // Possible return values for FindListToken.
  enum class ListTokenKind { Comma, Close, CommaClose };

  // Supported kinds for HandleBraceExpression.
  enum class BraceExpressionKind { Unknown, Value, Type };

  // Supported kinds for HandlePattern.
  enum class PatternKind { Parameter, Variable };

  // Gives information about the language construct/context being parsed. For
  // now, a simple enum but can be extended later to provide more information as
  // necessary.
  enum class ParseContext {
    File,  // Top-level context.
    Interface,
  };

  // Helper class for tracing state_stack_ on crashes.
  class PrettyStackTraceParseState;

  // Used to track state on state_stack_.
  struct StateStackEntry {
    StateStackEntry(ParserState state, PrecedenceGroup ambient_precedence,
                    PrecedenceGroup lhs_precedence,
                    TokenizedBuffer::Token token, int32_t subtree_start)
        : state(state),
          ambient_precedence(ambient_precedence),
          lhs_precedence(lhs_precedence),
          token(token),
          subtree_start(subtree_start) {}

    // Prints state information for verbose output.
    auto Print(llvm::raw_ostream& output) const -> void {
      output << state << " @" << token << " subtree_start=" << subtree_start
             << " has_error=" << has_error;
    };

    // The state.
    ParserState state;
    // Set to true to indicate that an error was found, and that contextual
    // error recovery may be needed.
    bool has_error = false;

    // Precedence information used by expression states in order to determine
    // operator precedence. The ambient_precedence deals with how the expression
    // should interact with outside context, while the lhs_precedence is
    // specific to the lhs of an operator expression.
    PrecedenceGroup ambient_precedence;
    PrecedenceGroup lhs_precedence;

    // A token providing context based on the subtree. This will typically be
    // the first token in the subtree, but may sometimes be a token within. It
    // will typically be used for the subtree's root node.
    TokenizedBuffer::Token token;
    // The offset within the ParseTree of the subtree start.
    int32_t subtree_start;
  };

  // We expect StateStackEntry to fit into 12 bytes:
  //   state = 1 byte
  //   has_error = 1 byte
  //   ambient_precedence = 1 byte
  //   lhs_precedence = 1 byte
  //   token = 4 bytes
  //   subtree_start = 4 bytes
  // If it becomes bigger, it'd be worth examining better packing; it should be
  // feasible to pack the 1-byte entries more tightly.
  static_assert(sizeof(StateStackEntry) == 12,
                "StateStackEntry has unexpected size!");

  Parser(ParseTree& tree, TokenizedBuffer& tokens,
         TokenDiagnosticEmitter& emitter, llvm::raw_ostream* vlog_stream);

  auto Parse() -> void;

  // Adds a node to the parse tree that has no children (a leaf).
  auto AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                   bool has_error = false) -> void;

  // Adds a node to the parse tree that has children.
  auto AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
               int subtree_start, bool has_error) -> void;

  // Returns the current position and moves past it.
  auto Consume() -> TokenizedBuffer::Token { return *(position_++); }

  // Parses an open paren token, possibly diagnosing if necessary. Creates a
  // leaf parse node of the specified start kind. The default_token is used when
  // there's no open paren.
  auto ConsumeAndAddOpenParen(TokenizedBuffer::Token default_token,
                              ParseNodeKind start_kind) -> void;

  // Parses a close paren token corresponding to the given open paren token,
  // possibly skipping forward and diagnosing if necessary. Creates a parse node
  // of the specified close kind.
  auto ConsumeAndAddCloseParen(StateStackEntry state, ParseNodeKind close_kind)
      -> void;

  // Composes `ConsumeIf` and `AddLeafNode`, returning false when ConsumeIf
  // fails.
  auto ConsumeAndAddLeafNodeIf(TokenKind token_kind, ParseNodeKind node_kind)
      -> bool;

  // Returns the current position and moves past it. Requires the token is the
  // expected kind.
  auto ConsumeChecked(TokenKind kind) -> TokenizedBuffer::Token;

  // If the current position's token matches this `Kind`, returns it and
  // advances to the next position. Otherwise returns an empty optional.
  auto ConsumeIf(TokenKind kind) -> std::optional<TokenizedBuffer::Token>;

  // Find the next token of any of the given kinds at the current bracketing
  // level.
  auto FindNextOf(std::initializer_list<TokenKind> desired_kinds)
      -> std::optional<TokenizedBuffer::Token>;

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
      -> std::optional<TokenizedBuffer::Token>;

  // Skip forward to the given token. Verifies that it is actually forward.
  auto SkipTo(TokenizedBuffer::Token t) -> void;

  // Returns true if the current token satisfies the lexical validity rules
  // for an infix operator.
  auto IsLexicallyValidInfixOperator() -> bool;

  // Determines whether the current trailing operator should be treated as
  // infix.
  auto IsTrailingOperatorInfix() -> bool;

  // Diagnoses whether the current token is not written properly for the given
  // fixity. For example, because mandatory whitespace is missing. Regardless of
  // whether there's an error, it's expected that parsing continues.
  auto DiagnoseOperatorFixity(OperatorFixity fixity) -> void;

  // If the current position is a `,`, consumes it, adds the provided token, and
  // returns `Comma`. Returns `Close` if the current position is close_token
  // (for example, `)`). `CommaClose` indicates it found both (for example,
  // `,)`). Handles cases where invalid tokens are present by advancing the
  // position, and may emit errors. Pass already_has_error in order to suppress
  // duplicate errors.
  auto ConsumeListToken(ParseNodeKind comma_kind, TokenKind close_kind,
                        bool already_has_error) -> ListTokenKind;

  // Gets the kind of the next token to be consumed.
  auto PositionKind() const -> TokenKind {
    return tokens_->GetKind(*position_);
  }

  // Tests whether the next token to be consumed is of the specified kind.
  auto PositionIs(TokenKind kind) const -> bool {
    return PositionKind() == kind;
  }

  // Pops the state and keeps the value for inspection.
  auto PopState() -> StateStackEntry {
    auto back = state_stack_.pop_back_val();
    CARBON_VLOG() << "Pop " << state_stack_.size() << ": " << back << "\n";
    return back;
  }

  // Pops the state and discards it.
  auto PopAndDiscardState() -> void {
    CARBON_VLOG() << "PopAndDiscard " << state_stack_.size() - 1 << ": "
                  << state_stack_.back() << "\n";
    state_stack_.pop_back();
  }

  // Pushes a new state with the current position for context.
  auto PushState(ParserState state) -> void {
    PushState(StateStackEntry(state, PrecedenceGroup::ForTopLevelExpression(),
                              PrecedenceGroup::ForTopLevelExpression(),
                              *position_, tree_->size()));
  }

  // Pushes a new expression state with specific precedence.
  auto PushStateForExpression(PrecedenceGroup ambient_precedence) -> void {
    PushState(StateStackEntry(ParserState::Expression, ambient_precedence,
                              PrecedenceGroup::ForTopLevelExpression(),
                              *position_, tree_->size()));
  }

  // Pushes a new state with detailed precedence for expression resume states.
  auto PushStateForExpressionLoop(ParserState state,
                                  PrecedenceGroup ambient_precedence,
                                  PrecedenceGroup lhs_precedence) -> void {
    PushState(StateStackEntry(state, ambient_precedence, lhs_precedence,
                              *position_, tree_->size()));
  }

  // Pushes a constructed state onto the stack.
  auto PushState(StateStackEntry state) -> void {
    CARBON_VLOG() << "Push " << state_stack_.size() << ": " << state << "\n";
    state_stack_.push_back(state);
    CARBON_CHECK(state_stack_.size() < (1 << 20))
        << "Excessive stack size: likely infinite loop";
  }

  // Propagates an error up the state stack, to the parent state.
  auto ReturnErrorOnState() -> void { state_stack_.back().has_error = true; }

  // Returns the appropriate ParserState for the input kind.
  static auto BraceExpressionKindToParserState(BraceExpressionKind kind,
                                               ParserState type,
                                               ParserState value,
                                               ParserState unknown)
      -> ParserState;

  // Prints a diagnostic for brace expression syntax errors.
  auto HandleBraceExpressionParameterError(StateStackEntry state,
                                           BraceExpressionKind kind) -> void;

  // Handles BraceExpressionParameterAs(Type|Value|Unknown).
  auto HandleBraceExpressionParameter(BraceExpressionKind kind) -> void;

  // Handles BraceExpressionParameterAfterDesignatorAs(Type|Value|Unknown).
  auto HandleBraceExpressionParameterAfterDesignator(BraceExpressionKind kind)
      -> void;

  // Handles BraceExpressionParameterFinishAs(Type|Value|Unknown).
  auto HandleBraceExpressionParameterFinish(BraceExpressionKind kind) -> void;

  // Handles BraceExpressionFinishAs(Type|Value|Unknown).
  auto HandleBraceExpressionFinish(BraceExpressionKind kind) -> void;

  // Handles DesignatorAs.
  auto HandleDesignator(bool as_struct) -> void;

  // When handling errors before the start of the definition, treat it as a
  // declaration. Recover to a semicolon when it makes sense as a possible
  // function end, otherwise use the fn token for the error.
  auto HandleFunctionError(StateStackEntry state, bool skip_past_likely_end)
      -> void;

  // Handles ParenConditionAs(If|While)
  auto HandleParenCondition(ParseNodeKind start_kind, ParserState finish_state)
      -> void;

  // Handles ParenExpressionParameterFinishAs(Unknown|Tuple).
  auto HandleParenExpressionParameterFinish(bool as_tuple) -> void;

  // Handles PatternAs(FunctionParameter|Variable).
  auto HandlePattern(PatternKind pattern_kind) -> void;

  // Handles the `;` after a keyword statement.
  auto HandleStatementKeywordFinish(ParseNodeKind node_kind) -> void;

  // Handles VarAs(Semicolon|For).
  auto HandleVar(ParserState finish_state) -> void;

  // `clang-format` has a bug with spacing around `->` returns in macros. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_PARSER_STATE(Name) auto Handle##Name##State()->void;
#include "toolchain/parser/parser_state.def"

  ParseTree* tree_;
  TokenizedBuffer* tokens_;
  TokenDiagnosticEmitter* emitter_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The current position within the token buffer.
  TokenizedBuffer::TokenIterator position_;
  // The EndOfFile token.
  TokenizedBuffer::TokenIterator end_;

  llvm::SmallVector<StateStackEntry> state_stack_;
  // TODO: This can be a mini-stack of contexts rather than a simple variable.
  ParseContext stack_context_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_H_

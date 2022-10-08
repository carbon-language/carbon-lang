// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_IMPL_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_IMPL_H_

#include "llvm/ADT/Optional.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/parser/precedence.h"

namespace Carbon {

class ParseTree::Parser {
 public:
  // Parses the tokens into a parse tree, emitting any errors encountered.
  //
  // This is the entry point to the parser implementation.
  static auto Parse(TokenizedBuffer& tokens, TokenDiagnosticEmitter& emitter)
      -> ParseTree;

 private:
  class ScopedStackStep;
  struct SubtreeStart;

  explicit Parser(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                  TokenDiagnosticEmitter& emitter);

  auto AtEndOfFile() -> bool {
    return tokens_.GetKind(*position_) == TokenKind::EndOfFile();
  }

  // Gets the kind of the next token to be consumed.
  [[nodiscard]] auto NextTokenKind() const -> TokenKind {
    return tokens_.GetKind(*position_);
  }

  // Tests whether the next token to be consumed is of the specified kind.
  [[nodiscard]] auto NextTokenIs(TokenKind kind) const -> bool {
    return NextTokenKind() == kind;
  }

  // Tests whether the next token to be consumed is of any of the specified
  // kinds.
  [[nodiscard]] auto NextTokenIsOneOf(
      std::initializer_list<TokenKind> kinds) const -> bool {
    return NextTokenKind().IsOneOf(kinds);
  }

  // Requires (and asserts) that the current position matches the provided
  // `Kind`. Returns the current token and advances to the next position.
  auto Consume(TokenKind kind) -> TokenizedBuffer::Token;

  // If the current position's token matches this `Kind`, returns it and
  // advances to the next position. Otherwise returns an empty optional.
  auto ConsumeIf(TokenKind kind) -> llvm::Optional<TokenizedBuffer::Token>;

  // Adds a node to the parse tree that is fully parsed, has no children
  // ("leaf"), and has a subsequent sibling.
  //
  // This sets up the next sibling of the node to be the next node in the parse
  // tree's preorder sequence.
  auto AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token) -> Node;

  // Composes `ConsumeIf` and `AddLeafNode`, propagating the failure case
  // through the optional.
  auto ConsumeAndAddLeafNodeIf(TokenKind t_kind, ParseNodeKind n_kind)
      -> llvm::Optional<Node>;

  // Marks the node `n` as having some parse errors and that the tree contains
  // a node with a parse error.
  auto MarkNodeError(Node n) -> void;

  // Tracks the current location as a potential start of a subtree.
  //
  // This returns a marker representing the current position, which can later
  // be used in a call to `AddNode` to mark all nodes created since this
  // position as children of the added node.
  auto GetSubtreeStartPosition() -> SubtreeStart;

  // Add a node to the parse tree that potentially has a subtree larger than
  // itself.
  //
  // Requires a start marker be passed to compute the size of the subtree rooted
  // at this node.
  auto AddNode(ParseNodeKind n_kind, TokenizedBuffer::Token t,
               SubtreeStart start, bool has_error = false) -> Node;

  // If the current token is an opening symbol for a matched group, skips
  // forward to one past the matched closing symbol and returns true. Otherwise,
  // returns false.
  auto SkipMatchingGroup() -> bool;

  // Skip forward to the given token.
  auto SkipTo(TokenizedBuffer::Token t) -> void;

  // Find the next token of any of the given kinds at the current bracketing
  // level.
  auto FindNextOf(std::initializer_list<TokenKind> desired_kinds)
      -> llvm::Optional<TokenizedBuffer::Token>;

  // Callback used if we find a semicolon when skipping to the end of a
  // declaration or statement.
  using SemiHandler = llvm::function_ref<
      auto(TokenizedBuffer::Token semi)->llvm::Optional<Node>>;

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
  // If we find a semicolon based on this skipping, we call `on_semi_` to try
  // to build a parse node to represent it, and will return that node.
  // Otherwise we will return an empty optional.
  auto SkipPastLikelyEnd(TokenizedBuffer::Token skip_root, SemiHandler on_semi)
      -> llvm::Optional<Node>;

  // Parses a close paren token corresponding to the given open paren token,
  // possibly skipping forward and diagnosing if necessary. Creates and returns
  // a parse node of the specified kind if successful.
  auto ParseCloseParen(TokenizedBuffer::Token open_paren, ParseNodeKind kind)
      -> llvm::Optional<Node>;

  // Parses a comma-separated list with the given delimiters.
  template <typename ListElementParser, typename ListCompletionHandler>
  auto ParseList(TokenKind open, TokenKind close,
                 ListElementParser list_element_parser,
                 ParseNodeKind comma_kind, ListCompletionHandler list_handler,
                 bool allow_trailing_comma = false) -> llvm::Optional<Node>;

  // Parses a parenthesized, comma-separated list.
  template <typename ListElementParser, typename ListCompletionHandler>
  auto ParseParenList(ListElementParser list_element_parser,
                      ParseNodeKind comma_kind,
                      ListCompletionHandler list_handler,
                      bool allow_trailing_comma = false)
      -> llvm::Optional<Node> {
    return ParseList(TokenKind::OpenParen(), TokenKind::CloseParen(),
                     list_element_parser, comma_kind, list_handler,
                     allow_trailing_comma);
  }

  // Parses a single function parameter declaration.
  auto ParseFunctionParameter() -> llvm::Optional<Node>;

  // Parses the signature of the function, consisting of a parameter list and an
  // optional return type. Returns the root node of the signature which must be
  // based on the open parenthesis of the parameter list.
  auto ParseFunctionSignature() -> bool;

  // Parses a block of code: `{ ... }`.
  //
  // These can form the definition for a function or be nested within a function
  // definition. These contain variable declarations and statements.
  auto ParseCodeBlock() -> llvm::Optional<Node>;

  // Parses a function declaration with an optional definition. Returns the
  // function parse node which is based on the `fn` introducer keyword.
  auto ParseFunctionDeclaration() -> Node;

  // Parses a variable declaration with an optional initializer.
  auto ParseVariableDeclaration() -> Node;

  // Parses and returns an empty declaration node from a single semicolon token.
  auto ParseEmptyDeclaration() -> Node;

  // Parses a package directive.
  auto ParsePackageDirective() -> Node;

  // Tries to parse a declaration. If a declaration, even an empty one after
  // skipping errors, can be parsed, it is returned. There may be parse errors
  // even when a node is returned.
  auto ParseDeclaration() -> llvm::Optional<Node>;

  // Parses a parenthesized expression.
  auto ParseParenExpression() -> llvm::Optional<Node>;

  // Parses a braced expression.
  auto ParseBraceExpression() -> llvm::Optional<Node>;

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  auto ParsePrimaryExpression() -> llvm::Optional<Node>;

  // Parses a designator expression suffix starting with `.`.
  auto ParseDesignatorExpression(SubtreeStart start, ParseNodeKind kind,
                                 bool has_errors) -> llvm::Optional<Node>;

  // Parses a call expression suffix starting with `(`.
  auto ParseCallExpression(SubtreeStart start, bool has_errors)
      -> llvm::Optional<Node>;

  // Parses a postfix expression, which is a primary expression followed by
  // zero or more of the following:
  //
  // -   function applications
  // -   array indexes (TODO)
  // -   designators
  auto ParsePostfixExpression() -> llvm::Optional<Node>;

  enum class OperatorFixity { Prefix, Infix, Postfix };

  // Determines whether the current token satisfies the lexical validity rules
  // for an infix operator.
  auto IsLexicallyValidInfixOperator() -> bool;

  // Diagnoses if the current token is not written properly for the given
  // fixity, for example because mandatory whitespace is missing.
  auto DiagnoseOperatorFixity(OperatorFixity fixity) -> void;

  // Determines whether the current trailing operator should be treated as
  // infix.
  auto IsTrailingOperatorInfix() -> bool;

  // Parses an expression involving operators, in a context with the given
  // precedence.
  auto ParseOperatorExpression(PrecedenceGroup precedence)
      -> llvm::Optional<Node>;

  // Parses an expression.
  auto ParseExpression() -> llvm::Optional<Node>;

  // Parses a type expression.
  auto ParseType() -> llvm::Optional<Node>;

  // Parses an expression statement: an expression followed by a semicolon.
  auto ParseExpressionStatement() -> llvm::Optional<Node>;

  // Parses the parenthesized condition in an if-statement.
  auto ParseParenCondition(TokenKind introducer) -> llvm::Optional<Node>;

  // Parses an if-statement.
  auto ParseIfStatement() -> llvm::Optional<Node>;

  // Parses a while-statement.
  auto ParseWhileStatement() -> llvm::Optional<Node>;

  // Parses a for-statement.
  auto ParseForStatement() -> llvm::Optional<Node>;

  enum class KeywordStatementArgument {
    None,
    Optional,
    Mandatory,
  };

  // Parses a statement of the form `keyword;` such as `break;` or `continue;`.
  auto ParseKeywordStatement(ParseNodeKind kind,
                             KeywordStatementArgument argument)
      -> llvm::Optional<Node>;

  // Parses a statement.
  auto ParseStatement() -> llvm::Optional<Node>;

  enum class PatternKind {
    Parameter,
    Variable,
  };

  // Parses a pattern.
  auto ParsePattern(PatternKind kind) -> llvm::Optional<Node>;

  ParseTree& tree_;
  TokenizedBuffer& tokens_;
  TokenDiagnosticEmitter& emitter_;

  // The current position within the token buffer. Never equal to `end`.
  TokenizedBuffer::TokenIterator position_;
  // The end position of the token buffer. There will always be an `EndOfFile`
  // token between `position` (inclusive) and `end` (exclusive).
  TokenizedBuffer::TokenIterator end_;

  // Managed through RETURN_IF_STACK_LIMITED, which should be invoked by all
  // functions.
  int stack_depth_ = 0;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_IMPL_H_

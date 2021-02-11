// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PARSER_PARSER_IMPL_H_
#define PARSER_PARSER_IMPL_H_

#include "diagnostics/diagnostic_emitter.h"
#include "lexer/token_kind.h"
#include "lexer/tokenized_buffer.h"
#include "llvm/ADT/Optional.h"
#include "parser/parse_node_kind.h"
#include "parser/parse_tree.h"

namespace Carbon {

class ParseTree::Parser {
 public:
  // Parses the tokens into a parse tree, emitting any errors encountered.
  //
  // This is the entry point to the parser implementation.
  static auto Parse(TokenizedBuffer& tokens, DiagnosticEmitter& de)
      -> ParseTree;

 private:
  struct SubtreeStart;

  explicit Parser(ParseTree& tree_arg, TokenizedBuffer& tokens_arg)
      : tree(tree_arg),
        tokens(tokens_arg),
        position(tokens.Tokens().begin()),
        end(tokens.Tokens().end()) {}

  // Requires (and asserts) that the current position matches the provide
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

  // Composes `consumeIf` and `addLeafNode`, propagating the failure case
  // through the optional.
  auto ConsumeAndAddLeafNodeIf(TokenKind t_kind, ParseNodeKind n_kind)
      -> llvm::Optional<Node>;

  // Marks the node `N` as having some parse error and that the tree contains
  // a node with a parse error.
  auto MarkNodeError(Node n) -> void;

  // Start parsing one (or more) subtrees of nodes.
  //
  // This returns a marker representing start position. It will also enforce
  // that at least *some* node is added using this starting position. Multiple
  // nodes can be added if they share a start position though.
  auto StartSubtree() -> SubtreeStart;

  // Add a node to the parse tree that potentially has a subtree larger than
  // itself.
  //
  // Requires a start marker be passed to compute the size of the subtree rooted
  // at this node.
  auto AddNode(ParseNodeKind n_kind, TokenizedBuffer::Token t,
               SubtreeStart& start, bool has_error = false) -> Node;

  // If the current token is an opening symbol for a matched group, skips
  // forward to one past the matched closing symbol and returns true. Otherwise,
  // returns false.
  auto SkipMatchingGroup() -> bool;

  // Skips forward to move past the likely end of a declaration.
  //
  // Looks forward, skipping over any matched symbol groups, to find the next
  // position that is likely past the end of a declaration. This is a heuristic
  // and should only be called when skipping past parse errors.
  //
  // The strategy for recognizing when we have likely passed the end of a
  // declaration:
  // - If we get to close curly brace, we likely ended the entire context of
  //   declarations.
  // - If we get to a semicolon, that should have ended the declaration.
  // - If we get to a new line from the `SkipRoot` token, but with the same or
  //   less indentation, there is likely a missing semicolon. Continued
  //   declarations across multiple lines should be indented.
  //
  // If we find a semicolon based on this skipping, we try to build a parse node
  // to represent it and will return that node. Otherwise we will return an
  // empty optional. If `IsInsideDeclaration` is true (the default) we build a
  // node that marks the end of the declaration we are inside. Otherwise we
  // build an empty declaration node.
  auto SkipPastLikelyDeclarationEnd(TokenizedBuffer::Token skip_root,
                                    bool is_inside_declaration = true)
      -> llvm::Optional<Node>;

  // Parses the signature of the function, consisting of a parameter list and an
  // optional return type. Returns the root node of the signature which must be
  // based on the open parenthesis of the parameter list.
  auto ParseFunctionSignature() -> Node;

  // Parses a block of code: `{ ... }`.
  //
  // These can form the definition for a function or be nested within a function
  // definition. These contain variable declarations and statements.
  auto ParseCodeBlock() -> Node;

  // Parses a function declaration with an optional definition. Returns the
  // function parse node which is based on the `fn` introducer keyword.
  auto ParseFunctionDeclaration() -> Node;

  // Parses and returns an empty declaration node from a single semicolon token.
  auto ParseEmptyDeclaration() -> Node;

  // Tries to parse a declaration. If a declaration, even an empty one after
  // skipping errors, can be parsed, it is returned. There may be parse errors
  // even when a node is returned.
  auto ParseDeclaration() -> llvm::Optional<Node>;

  ParseTree& tree;
  TokenizedBuffer& tokens;

  TokenizedBuffer::TokenIterator position;
  TokenizedBuffer::TokenIterator end;
};

}  // namespace Carbon

#endif  // PARSER_PARSER_IMPL_H_

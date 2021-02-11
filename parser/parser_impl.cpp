// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "parser/parser_impl.h"

#include <cstdlib>

#include "lexer/token_kind.h"
#include "lexer/tokenized_buffer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"
#include "parser/parse_node_kind.h"
#include "parser/parse_tree.h"

namespace Carbon {

auto ParseTree::Parser::Parse(TokenizedBuffer& tokens,
                              DiagnosticEmitter& /*unused*/) -> ParseTree {
  ParseTree tree(tokens);

  // We expect to have a 1:1 correspondence between tokens and tree nodes, so
  // reserve the space we expect to need here to avoid allocation and copying
  // overhead.
  tree.node_impls.reserve(tokens.Size());

  Parser parser(tree, tokens);
  while (parser.position != parser.end) {
    parser.ParseDeclaration();
  }

  assert(tree.Verify() && "Parse tree built but does not verify!");
  return tree;
}

auto ParseTree::Parser::Consume(TokenKind kind) -> TokenizedBuffer::Token {
  TokenizedBuffer::Token t = *position;
  assert(tokens.GetKind(t) == kind && "The current token is the wrong kind!");
  ++position;
  return t;
}

auto ParseTree::Parser::ConsumeIf(TokenKind kind)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (tokens.GetKind(*position) != kind) {
    return {};
  }

  return *position++;
}

auto ParseTree::Parser::AddLeafNode(ParseNodeKind kind,
                                    TokenizedBuffer::Token token) -> Node {
  Node n(tree.node_impls.size());
  tree.node_impls.push_back(NodeImpl(kind, token, /*subtree_size_arg=*/1));
  return n;
}

auto ParseTree::Parser::ConsumeAndAddLeafNodeIf(TokenKind t_kind,
                                                ParseNodeKind n_kind)
    -> llvm::Optional<Node> {
  auto t = ConsumeIf(t_kind);
  if (!t) {
    return {};
  }

  return AddLeafNode(n_kind, *t);
}

auto ParseTree::Parser::MarkNodeError(Node n) -> void {
  tree.node_impls[n.index].has_error = true;
  tree.has_errors = true;
}

// A marker for the start of a node's subtree.
//
// This is used to track the size of the node's subtree and ensure at least one
// parse node is added. It can be used repeatedly if multiple subtrees start at
// the same position.
struct ParseTree::Parser::SubtreeStart {
  int tree_size;
  bool node_added = false;

  ~SubtreeStart() {
    assert(node_added && "Never added a node for a subtree region!");
  }
};

auto ParseTree::Parser::StartSubtree() -> SubtreeStart {
  return {static_cast<int>(tree.node_impls.size())};
}

auto ParseTree::Parser::AddNode(ParseNodeKind n_kind, TokenizedBuffer::Token t,
                                SubtreeStart& start, bool has_error) -> Node {
  // The size of the subtree is the change in size from when we started this
  // subtree to now, but including the node we're about to add.
  int tree_stop_size = static_cast<int>(tree.node_impls.size()) + 1;
  int subtree_size = tree_stop_size - start.tree_size;

  Node n(tree.node_impls.size());
  tree.node_impls.push_back(NodeImpl(n_kind, t, subtree_size));
  if (has_error) {
    MarkNodeError(n);
  }

  start.node_added = true;
  return n;
}

auto ParseTree::Parser::SkipMatchingGroup() -> bool {
  assert(position != end && "Cannot skip at the end!");
  TokenizedBuffer::Token t = *position;
  TokenKind t_kind = tokens.GetKind(t);
  if (!t_kind.IsOpeningSymbol()) {
    return false;
  }

  position = std::next(
      TokenizedBuffer::TokenIterator(tokens.GetMatchedClosingToken(t)));
  return true;
}

auto ParseTree::Parser::SkipPastLikelyDeclarationEnd(
    TokenizedBuffer::Token skip_root, bool is_inside_declaration)
    -> llvm::Optional<Node> {
  if (position == end) {
    return {};
  }

  TokenizedBuffer::Line root_line = tokens.GetLine(skip_root);
  int root_line_indent = tokens.GetIndentColumnNumber(root_line);

  // We will keep scanning through tokens on the same line as the root or
  // lines with greater indentation than root's line.
  auto is_same_line_or_indent_greater_than_root =
      [&](TokenizedBuffer::Token t) {
        TokenizedBuffer::Line l = tokens.GetLine(t);
        if (l == root_line) {
          return true;
        }

        return tokens.GetIndentColumnNumber(l) > root_line_indent;
      };

  do {
    TokenKind current_kind = tokens.GetKind(*position);
    if (current_kind == TokenKind::CloseCurlyBrace()) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return {};
    }

    // If we find a semicolon, we want to parse it to end the declaration.
    if (current_kind == TokenKind::Semi()) {
      TokenizedBuffer::Token semi = *position++;

      // Add a node for the semicolon. If we're inside of a declaration, this
      // is a declaration ending semicolon, otherwise it simply forms an empty
      // declaration.
      return AddLeafNode(is_inside_declaration
                             ? ParseNodeKind::DeclarationEnd()
                             : ParseNodeKind::EmptyDeclaration(),
                         semi);
    }

    // Skip over any matching group of tokens.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position;
  } while (position != end &&
           is_same_line_or_indent_greater_than_root(*position));

  return {};
}

auto ParseTree::Parser::ParseFunctionSignature() -> Node {
  assert(position != end && "Cannot parse past the end!");

  TokenizedBuffer::Token open_paren = Consume(TokenKind::OpenParen());
  assert(position != end &&
         "The lexer ensures we always have a closing paren!");
  auto start = StartSubtree();

  // FIXME: Add support for parsing parameters.

  bool has_errors = false;
  auto close_paren = ConsumeIf(TokenKind::CloseParen());
  if (!close_paren) {
    llvm::errs() << "ERROR: unexpected token before the close of the "
                    "parameters on line "
                 << tokens.GetLineNumber(*position) << "!\n";
    has_errors = true;

    // We can trivially skip to the actual close parenthesis from here.
    close_paren = tokens.GetMatchedClosingToken(open_paren);
    position = std::next(TokenizedBuffer::TokenIterator(*close_paren));
  }
  AddLeafNode(ParseNodeKind::ParameterListEnd(), *close_paren);

  // FIXME: Implement parsing of a return type.

  return AddNode(ParseNodeKind::ParameterList(), open_paren, start, has_errors);
}

struct ParseTree::Parser::CodeBlock {
  CodeBlock(TokenizedBuffer::Token in_open_curly, SubtreeStart in_start)
      : open_curly(in_open_curly), start(in_start) {}

  TokenizedBuffer::Token open_curly;
  SubtreeStart start;
  bool has_errors = false;
};

auto ParseTree::Parser::StartCodeBlock() -> CodeBlock {
  TokenizedBuffer::Token open_curly = Consume(TokenKind::OpenCurlyBrace());
  assert(position != end &&
         "The lexer ensures we always have a closing curly!");
  return CodeBlock(open_curly, StartSubtree());
}

auto ParseTree::Parser::ParseCodeBlock() -> Node {
  assert(position != end && "Cannot parse past the end!");

  llvm::SmallVector<CodeBlock, 16> block_stack;
  block_stack.push_back(StartCodeBlock());

  // Loop until a matching close curly is found.
  for (;;) {
    switch (tokens.GetKind(*position)) {
      case TokenKind::OpenCurlyBrace():
        block_stack.push_back(StartCodeBlock());
        break;

      case TokenKind::CloseCurlyBrace(): {
        auto block = block_stack.pop_back_val();
        // We always reach here having set our position in the token stream to
        // the close curly brace.
        AddLeafNode(ParseNodeKind::CodeBlockEnd(),
                    Consume(TokenKind::CloseCurlyBrace()));

        auto node = AddNode(ParseNodeKind::CodeBlock(), block.open_curly,
                            block.start, block.has_errors);
        if (block_stack.empty()) {
          return node;
        }
        break;
      }

      default: {
        // FIXME: Add support for parsing more expressions & statements.
        llvm::errs() << "ERROR: unexpected token before the close of the "
                        "function definition on line "
                     << tokens.GetLineNumber(*position) << "!\n";
        auto& block = *block_stack.rbegin();
        block.has_errors = true;

        // We can trivially skip to the actual close curly brace from here.
        position = TokenizedBuffer::TokenIterator(
            tokens.GetMatchedClosingToken(block.open_curly));
        break;
      }
    }
  }
}

auto ParseTree::Parser::ParseFunctionDeclaration() -> Node {
  assert(position != end && "Cannot parse past the end!");

  TokenizedBuffer::Token function_intro_token = Consume(TokenKind::FnKeyword());
  auto start = StartSubtree();
  auto add_error_function_node = [&] {
    return AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
                   start, /*has_error=*/true);
  };

  if (position == end) {
    llvm::errs() << "ERROR: File ended with a function introducer on line "
                 << tokens.GetLineNumber(function_intro_token) << "!\n";
    return add_error_function_node();
  }

  auto name_n = ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                                        ParseNodeKind::Identifier());
  if (!name_n) {
    llvm::errs() << "ERROR: Function declaration with no name on line "
                 << tokens.GetLineNumber(function_intro_token) << "!\n";
    // FIXME: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    SkipPastLikelyDeclarationEnd(function_intro_token);
    return add_error_function_node();
  }
  if (position == end) {
    llvm::errs() << "ERROR: File ended after a function introducer and "
                    "identifier on line "
                 << tokens.GetLineNumber(function_intro_token) << "!\n";
    return add_error_function_node();
  }

  TokenizedBuffer::Token open_paren = *position;
  if (tokens.GetKind(open_paren) != TokenKind::OpenParen()) {
    llvm::errs()
        << "ERROR: Missing open parentheses in declaration of function '"
        << tokens.GetTokenText(tree.GetNodeToken(*name_n)) << "' on line "
        << tokens.GetLineNumber(function_intro_token) << "!\n";
    SkipPastLikelyDeclarationEnd(function_intro_token);
    return add_error_function_node();
  }
  assert(std::next(position) != end &&
         "Unbalanced parentheses should be rejected by the lexer.");
  TokenizedBuffer::Token close_paren =
      tokens.GetMatchedClosingToken(open_paren);

  Node signature_n = ParseFunctionSignature();
  assert(*std::prev(position) == close_paren &&
         "Should have parsed through the close paren, whether successfully "
         "or with errors.");
  if (tree.node_impls[signature_n.index].has_error) {
    // Don't try to parse more of the function declaration, but consume a
    // declaration ending semicolon if found (without going to a new line).
    SkipPastLikelyDeclarationEnd(function_intro_token);
    return add_error_function_node();
  }

  // See if we should parse a definition which is represented as a code block.
  if (tokens.GetKind(*position) == TokenKind::OpenCurlyBrace()) {
    ParseCodeBlock();
  } else if (!ConsumeAndAddLeafNodeIf(TokenKind::Semi(),
                                      ParseNodeKind::DeclarationEnd())) {
    llvm::errs() << "ERROR: Function declaration not terminated by a "
                    "semicolon on line "
                 << tokens.GetLineNumber(close_paren) << "!\n";
    if (tokens.GetLine(*position) == tokens.GetLine(close_paren)) {
      // Only need to skip if we've not already found a new line.
      SkipPastLikelyDeclarationEnd(function_intro_token);
    }
    return add_error_function_node();
  }

  // Successfully parsed the function, add that node.
  return AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
                 start);
}

auto ParseTree::Parser::ParseEmptyDeclaration() -> Node {
  assert(position != end && "Cannot parse past the end!");
  return AddLeafNode(ParseNodeKind::EmptyDeclaration(),
                     Consume(TokenKind::Semi()));
}

auto ParseTree::Parser::ParseDeclaration() -> llvm::Optional<Node> {
  assert(position != end && "Cannot parse past the end!");
  TokenizedBuffer::Token t = *position;
  switch (tokens.GetKind(t)) {
    case TokenKind::FnKeyword():
      return ParseFunctionDeclaration();
    case TokenKind::Semi():
      return ParseEmptyDeclaration();
    default:
      // Errors are handled outside the switch.
      break;
  }

  // We didn't recognize an introducer for a valid declaration.
  llvm::errs() << "ERROR: Unrecognized declaration introducer '"
               << tokens.GetTokenText(t) << "' on line "
               << tokens.GetLineNumber(t) << "!\n";

  // Skip forward past any end of a declaration we simply didn't understand so
  // that we can find the start of the next declaration or the end of a scope.
  if (auto found_semi_n =
          SkipPastLikelyDeclarationEnd(t, /*is_inside_declaration=*/false)) {
    MarkNodeError(*found_semi_n);
    return *found_semi_n;
  }

  // Nothing, not even a semicolon found. We still need to mark that an error
  // occurred though.
  tree.has_errors = true;
  return {};
}

}  // namespace Carbon

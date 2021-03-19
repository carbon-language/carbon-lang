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

struct UnexpectedTokenInFunctionParams
    : SimpleDiagnostic<UnexpectedTokenInFunctionParams> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unexpected token in function parameter list.";
};

struct UnexpectedTokenInCodeBlock
    : SimpleDiagnostic<UnexpectedTokenInCodeBlock> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unexpected token in code block.";
};

struct ExpectedFunctionName : SimpleDiagnostic<ExpectedFunctionName> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected function name after `fn` keyword.";
};

struct ExpectedFunctionParams : SimpleDiagnostic<ExpectedFunctionParams> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected `(` after function name.";
};

struct ExpectedFunctionBodyOrSemi
    : SimpleDiagnostic<ExpectedFunctionBodyOrSemi> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected function definition or `;` after function declaration.";
};

struct UnrecognizedDeclaration : SimpleDiagnostic<UnrecognizedDeclaration> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unrecognized declaration introducer.";
};

auto ParseTree::Parser::Parse(TokenizedBuffer& tokens,
                              TokenDiagnosticEmitter& emitter) -> ParseTree {
  ParseTree tree(tokens);

  // We expect to have a 1:1 correspondence between tokens and tree nodes, so
  // reserve the space we expect to need here to avoid allocation and copying
  // overhead.
  tree.node_impls.reserve(tokens.Size());

  Parser parser(tree, tokens, emitter);
  while (!parser.AtEndOfFile()) {
    parser.ParseDeclaration();
  }

  parser.AddLeafNode(ParseNodeKind::FileEnd(), *parser.position);

  assert(tree.Verify() && "Parse tree built but does not verify!");
  return tree;
}

ParseTree::Parser::Parser(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                          TokenDiagnosticEmitter& emitter)
    : tree(tree_arg),
      tokens(tokens_arg),
      emitter(emitter),
      position(tokens.Tokens().begin()),
      end(tokens.Tokens().end()) {
  assert(std::find_if(position, end,
                      [&](TokenizedBuffer::Token t) {
                        return tokens.GetKind(t) == TokenKind::EndOfFile();
                      }) != end &&
         "No EndOfFileToken in token buffer.");
}

auto ParseTree::Parser::Consume(TokenKind kind) -> TokenizedBuffer::Token {
  TokenizedBuffer::Token t = *position;
  assert(kind != TokenKind::EndOfFile() && "Cannot consume the EOF token!");
  assert(tokens.GetKind(t) == kind && "The current token is the wrong kind!");
  ++position;
  assert(position != end && "Reached end of tokens without finding EOF token.");
  return t;
}

auto ParseTree::Parser::ConsumeIf(TokenKind kind)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (tokens.GetKind(*position) != kind) {
    return {};
  }
  return Consume(kind);
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
  TokenizedBuffer::Token t = *position;
  TokenKind t_kind = tokens.GetKind(t);
  if (!t_kind.IsOpeningSymbol()) {
    return false;
  }

  SkipTo(tokens.GetMatchedClosingToken(t));
  Consume(t_kind.GetClosingSymbol());
  return true;
}

auto ParseTree::Parser::SkipTo(TokenizedBuffer::Token t) -> void {
  assert(t >= *position && "Tried to skip backwards.");
  position = TokenizedBuffer::TokenIterator(t);
  assert(position != end && "Skipped past EOF.");
}

auto ParseTree::Parser::SkipPastLikelyDeclarationEnd(
    TokenizedBuffer::Token skip_root, bool is_inside_declaration)
    -> llvm::Optional<Node> {
  if (AtEndOfFile()) {
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

    // If we find a semicolon, parse it and add a corresponding node. If we're
    // inside of a declaration, this is a declaration ending semicolon,
    // otherwise it simply forms an empty declaration.
    if (auto end_node = ConsumeAndAddLeafNodeIf(
            TokenKind::Semi(), is_inside_declaration
                                   ? ParseNodeKind::DeclarationEnd()
                                   : ParseNodeKind::EmptyDeclaration())) {
      return end_node;
    }

    // Skip over any matching group of tokens.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    Consume(current_kind);
  } while (!AtEndOfFile() &&
           is_same_line_or_indent_greater_than_root(*position));

  return {};
}

auto ParseTree::Parser::ParseFunctionSignature() -> Node {
  TokenizedBuffer::Token open_paren = Consume(TokenKind::OpenParen());
  auto start = StartSubtree();

  // FIXME: Add support for parsing parameters.

  bool has_errors = false;
  if (tokens.GetKind(*position) != TokenKind::CloseParen()) {
    emitter.EmitError<UnexpectedTokenInFunctionParams>(*position);
    has_errors = true;

    // We can trivially skip to the actual close parenthesis from here.
    SkipTo(tokens.GetMatchedClosingToken(open_paren));
  }
  AddLeafNode(ParseNodeKind::ParameterListEnd(),
              Consume(TokenKind::CloseParen()));

  // FIXME: Implement parsing of a return type.

  return AddNode(ParseNodeKind::ParameterList(), open_paren, start, has_errors);
}

auto ParseTree::Parser::ParseCodeBlock() -> Node {
  TokenizedBuffer::Token open_curly = Consume(TokenKind::OpenCurlyBrace());
  auto start = StartSubtree();

  bool has_errors = false;

  // Loop over all the different possibly nested elements in the code block.
  for (;;) {
    switch (tokens.GetKind(*position)) {
      default:
        // FIXME: Add support for parsing more expressions & statements.
        emitter.EmitError<UnexpectedTokenInCodeBlock>(*position);
        has_errors = true;

        // We can trivially skip to the actual close curly brace from here.
        SkipTo(tokens.GetMatchedClosingToken(open_curly));
        // Now fall through to the close curly brace handling code.
        LLVM_FALLTHROUGH;

      case TokenKind::CloseCurlyBrace():
        break;

      case TokenKind::OpenCurlyBrace():
        // FIXME: We should consider avoiding recursion here with some side
        // stack.
        ParseCodeBlock();
        continue;
    }

    // We only continue looping with `continue` above.
    break;
  }

  // We always reach here having set our position in the token stream to the
  // close curly brace.
  AddLeafNode(ParseNodeKind::CodeBlockEnd(),
              Consume(TokenKind::CloseCurlyBrace()));

  return AddNode(ParseNodeKind::CodeBlock(), open_curly, start, has_errors);
}

auto ParseTree::Parser::ParseFunctionDeclaration() -> Node {
  TokenizedBuffer::Token function_intro_token = Consume(TokenKind::FnKeyword());
  auto start = StartSubtree();

  auto add_error_function_node = [&] {
    return AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
                   start, /*has_error=*/true);
  };

  auto name_n = ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                                        ParseNodeKind::Identifier());
  if (!name_n) {
    emitter.EmitError<ExpectedFunctionName>(*position);
    // FIXME: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    SkipPastLikelyDeclarationEnd(function_intro_token);
    return add_error_function_node();
  }

  TokenizedBuffer::Token open_paren = *position;
  if (tokens.GetKind(open_paren) != TokenKind::OpenParen()) {
    emitter.EmitError<ExpectedFunctionParams>(open_paren);
    SkipPastLikelyDeclarationEnd(function_intro_token);
    return add_error_function_node();
  }
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
    emitter.EmitError<ExpectedFunctionBodyOrSemi>(*position);
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
  return AddLeafNode(ParseNodeKind::EmptyDeclaration(),
                     Consume(TokenKind::Semi()));
}

auto ParseTree::Parser::ParseDeclaration() -> llvm::Optional<Node> {
  TokenizedBuffer::Token t = *position;
  switch (tokens.GetKind(t)) {
    case TokenKind::FnKeyword():
      return ParseFunctionDeclaration();
    case TokenKind::Semi():
      return ParseEmptyDeclaration();
    case TokenKind::EndOfFile():
      return llvm::None;
    default:
      // Errors are handled outside the switch.
      break;
  }

  // We didn't recognize an introducer for a valid declaration.
  emitter.EmitError<UnrecognizedDeclaration>(t);

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

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

struct ExpectedVariableName : SimpleDiagnostic<ExpectedVariableName> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected variable name after type in `var` declaration.";
};

struct UnrecognizedDeclaration : SimpleDiagnostic<UnrecognizedDeclaration> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unrecognized declaration introducer.";
};

struct ExpectedExpression : SimpleDiagnostic<ExpectedExpression> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message = "Expected expression.";
};

struct ExpectedCloseParen : SimpleDiagnostic<ExpectedCloseParen> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unexpected tokens before `)`.";
};

struct ExpectedSemiAfterExpression
    : SimpleDiagnostic<ExpectedSemiAfterExpression> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected `;` after expression.";
};

struct ExpectedIdentifierAfterDot
    : SimpleDiagnostic<ExpectedIdentifierAfterDot> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected identifier after `.`.";
};

struct UnexpectedTokenInFunctionArgs
    : SimpleDiagnostic<UnexpectedTokenInFunctionArgs> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unexpected token in function argument list.";
};

struct OperatorRequiresParentheses
    : SimpleDiagnostic<OperatorRequiresParentheses> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Parentheses are required to disambiguate operator precedence.";
};

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

auto ParseTree::Parser::Parse(TokenizedBuffer& tokens,
                              TokenDiagnosticEmitter& emitter) -> ParseTree {
  ParseTree tree(tokens);

  // We expect to have a 1:1 correspondence between tokens and tree nodes, so
  // reserve the space we expect to need here to avoid allocation and copying
  // overhead.
  tree.node_impls.reserve(tokens.Size());

  Parser parser(tree, tokens, emitter);
  while (!parser.AtEndOfFile()) {
    if (!parser.ParseDeclaration()) {
      // We don't have an enclosing parse tree node to mark as erroneous, so
      // just mark the tree as a whole.
      tree.has_errors = true;
    }
  }

  parser.AddLeafNode(ParseNodeKind::FileEnd(), *parser.position);

  assert(tree.Verify() && "Parse tree built but does not verify!");
  return tree;
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
// This is used to track the size of the node's subtree. It can be used
// repeatedly if multiple subtrees start at the same position.
struct ParseTree::Parser::SubtreeStart {
  int tree_size;
};

auto ParseTree::Parser::StartSubtree() -> SubtreeStart {
  return {static_cast<int>(tree.node_impls.size())};
}

auto ParseTree::Parser::AddNode(ParseNodeKind n_kind, TokenizedBuffer::Token t,
                                SubtreeStart start, bool has_error) -> Node {
  // The size of the subtree is the change in size from when we started this
  // subtree to now, but including the node we're about to add.
  int tree_stop_size = static_cast<int>(tree.node_impls.size()) + 1;
  int subtree_size = tree_stop_size - start.tree_size;

  Node n(tree.node_impls.size());
  tree.node_impls.push_back(NodeImpl(n_kind, t, subtree_size));
  if (has_error) {
    MarkNodeError(n);
  }

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

auto ParseTree::Parser::FindNextOf(
    std::initializer_list<TokenKind> desired_kinds)
    -> llvm::Optional<TokenizedBuffer::Token> {
  auto new_position = position;
  while (true) {
    TokenizedBuffer::Token token = *new_position;
    TokenKind kind = tokens.GetKind(token);
    for (TokenKind desired_kind : desired_kinds) {
      if (kind == desired_kind) {
        return token;
      }
    }

    // Step to the next token at the current bracketing level.
    if (kind.IsClosingSymbol() || kind == TokenKind::EndOfFile()) {
      // There are no more tokens at this level.
      return llvm::None;
    } else if (kind.IsOpeningSymbol()) {
      new_position =
          TokenizedBuffer::TokenIterator(tokens.GetMatchedClosingToken(token));
    } else {
      ++new_position;
    }
  }
}

auto ParseTree::Parser::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root,
                                          SemiHandler on_semi)
    -> llvm::Optional<Node> {
  if (AtEndOfFile()) {
    return llvm::None;
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
      return llvm::None;
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(TokenKind::Semi())) {
      return on_semi(*semi);
    }

    // Skip over any matching group of tokens.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    Consume(current_kind);
  } while (!AtEndOfFile() &&
           is_same_line_or_indent_greater_than_root(*position));

  return llvm::None;
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
  while (tokens.GetKind(*position) != TokenKind::CloseCurlyBrace()) {
    if (!ParseStatement()) {
      // We detected and diagnosed an error of some kind. We can trivially skip
      // to the actual close curly brace from here.
      // FIXME: It would be better to skip to the next semicolon, or the next
      // token at the start of a line with the same indent as this one.
      SkipTo(tokens.GetMatchedClosingToken(open_curly));
      has_errors = true;
      break;
    }
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

  auto handle_semi_in_error_recovery = [&](TokenizedBuffer::Token semi) {
    return AddLeafNode(ParseNodeKind::DeclarationEnd(), semi);
  };

  auto name_n = ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                                        ParseNodeKind::DeclaredName());
  if (!name_n) {
    emitter.EmitError<ExpectedFunctionName>(*position);
    // FIXME: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    return add_error_function_node();
  }

  TokenizedBuffer::Token open_paren = *position;
  if (tokens.GetKind(open_paren) != TokenKind::OpenParen()) {
    emitter.EmitError<ExpectedFunctionParams>(open_paren);
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
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
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
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
      SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    }
    return add_error_function_node();
  }

  // Successfully parsed the function, add that node.
  return AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
                 start);
}

auto ParseTree::Parser::ParseVariableDeclaration() -> Node {
  // `var` expression identifier [= expression] `;`
  TokenizedBuffer::Token var_token = Consume(TokenKind::VarKeyword());
  auto start = StartSubtree();

  auto type = ParseExpression();

  auto name = ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                                      ParseNodeKind::DeclaredName());
  if (!name) {
    emitter.EmitError<ExpectedVariableName>(*position);
    if (auto after_name = FindNextOf({TokenKind::Equal(), TokenKind::Semi()})) {
      SkipTo(*after_name);
    }
  }

  auto start_init = StartSubtree();
  if (auto equal_token = ConsumeIf(TokenKind::Equal())) {
    auto init = ParseExpression();
    AddNode(ParseNodeKind::VariableInitializer(), *equal_token, start_init,
            /*has_error=*/!init);
  }

  auto semi = ConsumeAndAddLeafNodeIf(TokenKind::Semi(),
                                      ParseNodeKind::DeclarationEnd());
  if (!semi) {
    SkipPastLikelyEnd(var_token, [&](TokenizedBuffer::Token semi) {
      return AddLeafNode(ParseNodeKind::DeclarationEnd(), semi);
    });
  }

  return AddNode(ParseNodeKind::VariableDeclaration(), var_token, start,
                 /*has_error=*/!type || !name || !semi);
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
    case TokenKind::VarKeyword():
      return ParseVariableDeclaration();
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
          SkipPastLikelyEnd(t, [&](TokenizedBuffer::Token semi) {
            return AddLeafNode(ParseNodeKind::EmptyDeclaration(), semi);
          })) {
    MarkNodeError(*found_semi_n);
    return *found_semi_n;
  }

  // Nothing, not even a semicolon found.
  return llvm::None;
}

auto ParseTree::Parser::ParseParenExpression() -> llvm::Optional<Node> {
  // `(` expression `)`
  auto start = StartSubtree();
  TokenizedBuffer::Token open_paren = Consume(TokenKind::OpenParen());

  // TODO: If the next token is a close paren, build an empty tuple literal.

  bool has_errors = !ParseExpression();

  // TODO: If the next token is a comma, build a tuple literal.

  if (tokens.GetKind(*position) != TokenKind::CloseParen()) {
    if (!has_errors) {
      emitter.EmitError<ExpectedCloseParen>(*position);
      has_errors = true;
    }
    SkipTo(tokens.GetMatchedClosingToken(open_paren));
  }

  AddLeafNode(ParseNodeKind::ParenExpressionEnd(),
              Consume(TokenKind::CloseParen()));
  return AddNode(ParseNodeKind::ParenExpression(), open_paren, start,
                 has_errors);
}

auto ParseTree::Parser::ParsePrimaryExpression() -> llvm::Optional<Node> {
  TokenizedBuffer::Token t = *position;
  TokenKind token_kind = tokens.GetKind(t);
  llvm::Optional<ParseNodeKind> kind;
  switch (token_kind) {
    case TokenKind::Identifier():
      kind = ParseNodeKind::NameReference();
      break;

    case TokenKind::IntegerLiteral():
    case TokenKind::RealLiteral():
    case TokenKind::StringLiteral():
      kind = ParseNodeKind::Literal();
      break;

    case TokenKind::OpenParen():
      return ParseParenExpression();

    default:
      emitter.EmitError<ExpectedExpression>(t);
      return llvm::None;
  }

  return AddLeafNode(*kind, Consume(token_kind));
}

auto ParseTree::Parser::ParseDesignatorExpression(SubtreeStart start,
                                                  bool has_errors)
    -> llvm::Optional<Node> {
  // `.` identifier
  auto dot = Consume(TokenKind::Period());
  auto name = ConsumeIf(TokenKind::Identifier());
  if (name) {
    AddLeafNode(ParseNodeKind::DesignatedName(), *name);
  } else {
    // If we see a keyword, assume it was intended to be the designated name.
    // TODO: Should keywords be valid in designators?
    if (tokens.GetKind(*position).IsKeyword()) {
      Consume(tokens.GetKind(*position));
    }
    emitter.EmitError<ExpectedIdentifierAfterDot>(*position);
    has_errors = true;
  }
  return AddNode(ParseNodeKind::DesignatorExpression(), dot, start, has_errors);
}

auto ParseTree::Parser::ParseCallExpression(SubtreeStart start, bool has_errors)
    -> llvm::Optional<Node> {
  // `(` expression-list[opt] `)`
  //
  // expression-list ::= expression
  //                 ::= expression `,` expression-list
  TokenizedBuffer::Token open_paren = Consume(TokenKind::OpenParen());

  // Parse arguments, if any are specified.
  if (tokens.GetKind(*position) != TokenKind::CloseParen()) {
    while (true) {
      bool argument_error = !ParseExpression();
      has_errors |= argument_error;

      if (tokens.GetKind(*position) == TokenKind::CloseParen()) {
        break;
      }

      if (tokens.GetKind(*position) != TokenKind::Comma()) {
        if (!argument_error) {
          emitter.EmitError<UnexpectedTokenInFunctionArgs>(*position);
        }
        has_errors = true;

        auto comma_position = FindNextOf({TokenKind::Comma()});
        if (!comma_position) {
          SkipTo(tokens.GetMatchedClosingToken(open_paren));
          break;
        }
        SkipTo(*comma_position);
      }

      AddLeafNode(ParseNodeKind::CallExpressionComma(),
                  Consume(TokenKind::Comma()));
    }
  }

  AddLeafNode(ParseNodeKind::CallExpressionEnd(),
              Consume(TokenKind::CloseParen()));
  return AddNode(ParseNodeKind::CallExpression(), open_paren, start,
                 has_errors);
}

auto ParseTree::Parser::ParsePostfixExpression() -> llvm::Optional<Node> {
  auto start = StartSubtree();
  llvm::Optional<Node> expression = ParsePrimaryExpression();

  while (true) {
    switch (tokens.GetKind(*position)) {
      case TokenKind::Period():
        expression = ParseDesignatorExpression(start, !expression);
        break;

      case TokenKind::OpenParen():
        expression = ParseCallExpression(start, !expression);
        break;

      default: {
        return expression;
      }
    }
  }
}

auto ParseTree::Parser::ParseOperatorExpression(
    PrecedenceGroup ambient_precedence) -> llvm::Optional<Node> {
  auto start = StartSubtree();

  llvm::Optional<Node> lhs;
  PrecedenceGroup lhs_precedence = PrecedenceGroup::ForPostfixExpression();

  // Check for a prefix operator.
  if (auto operator_precedence =
          PrecedenceGroup::ForLeading(tokens.GetKind(*position));
      !operator_precedence) {
    lhs = ParsePostfixExpression();
  } else {
    if (PrecedenceGroup::GetPriority(ambient_precedence,
                                     *operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this prefix operator in this
      // context. Diagnose this, but carry on and parse it anyway.
      emitter.EmitError<OperatorRequiresParentheses>(*position);
    }

    auto operator_token = Consume(tokens.GetKind(*position));
    bool has_errors = !ParseOperatorExpression(*operator_precedence);
    lhs = AddNode(ParseNodeKind::PrefixOperator(), operator_token, start,
                  has_errors);
    lhs_precedence = *operator_precedence;
  }

  // Consume a sequence of infix and postfix operators.
  while (auto trailing_operator =
             PrecedenceGroup::ForTrailing(tokens.GetKind(*position))) {
    auto [operator_precedence, is_binary] = *trailing_operator;
    if (PrecedenceGroup::GetPriority(ambient_precedence, operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this operator in this context. Try
      // again in the enclosing expression context.
      return lhs;
    }

    if (PrecedenceGroup::GetPriority(lhs_precedence, operator_precedence) !=
        OperatorPriority::LeftFirst) {
      // Either the LHS operator and this operator are ambiguous, or the
      // LHS operaor is a unary operator that can't be nested within
      // this operator. Either way, parentheses are required.
      emitter.EmitError<OperatorRequiresParentheses>(*position);
      lhs = llvm::None;
    }

    auto operator_token = Consume(tokens.GetKind(*position));

    if (is_binary) {
      auto rhs = ParseOperatorExpression(operator_precedence);
      lhs = AddNode(ParseNodeKind::InfixOperator(), operator_token, start,
                    /*has_error=*/!lhs || !rhs);
    } else {
      lhs = AddNode(ParseNodeKind::PostfixOperator(), operator_token, start,
                    /*has_error=*/!lhs);
    }
    lhs_precedence = operator_precedence;
  }

  return lhs;
}

auto ParseTree::Parser::ParseExpression() -> llvm::Optional<Node> {
  return ParseOperatorExpression(PrecedenceGroup::ForTopLevelExpression());
}

auto ParseTree::Parser::ParseExpressionStatement() -> llvm::Optional<Node> {
  TokenizedBuffer::Token start_token = *position;
  auto start = StartSubtree();

  bool has_errors = !ParseExpression();

  if (auto semi = ConsumeIf(TokenKind::Semi())) {
    return AddNode(ParseNodeKind::ExpressionStatement(), *semi, start,
                   has_errors);
  }

  if (!has_errors) {
    emitter.EmitError<ExpectedSemiAfterExpression>(*position);
  }

  if (auto recovery_node =
          SkipPastLikelyEnd(start_token, [&](TokenizedBuffer::Token semi) {
            return AddNode(ParseNodeKind::ExpressionStatement(), semi, start,
                           true);
          })) {
    return recovery_node;
  }

  // Found junk not even followed by a `;`.
  return llvm::None;
}

auto ParseTree::Parser::ParseStatement() -> llvm::Optional<Node> {
  switch (tokens.GetKind(*position)) {
    case TokenKind::VarKeyword():
      return ParseVariableDeclaration();

    case TokenKind::OpenCurlyBrace():
      return ParseCodeBlock();

    default:
      // A statement with no introducer token can only be an expression
      // statement.
      return ParseExpressionStatement();
  }
}

}  // namespace Carbon

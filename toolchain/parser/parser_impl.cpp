// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_impl.h"

#include <cstdlib>

#include "common/check.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

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
      "Expected pattern in `var` declaration.";
};

struct ExpectedParameterName : SimpleDiagnostic<ExpectedParameterName> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected parameter declaration.";
};

struct ExpectedStructLiteralField
    : SimpleDiagnostic<ExpectedStructLiteralField> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";

  auto Format() -> std::string {
    std::string result = "Expected ";
    if (can_be_type) {
      result += "`.field: type`";
    }
    if (can_be_type && can_be_value) {
      result += " or ";
    }
    if (can_be_value) {
      result += "`.field = value`";
    }
    result += ".";
    return result;
  }

  bool can_be_type;
  bool can_be_value;
};

struct UnrecognizedDeclaration : SimpleDiagnostic<UnrecognizedDeclaration> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unrecognized declaration introducer.";
};

struct ExpectedCodeBlock : SimpleDiagnostic<ExpectedCodeBlock> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message = "Expected braced code block.";
};

struct ExpectedExpression : SimpleDiagnostic<ExpectedExpression> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message = "Expected expression.";
};

struct ExpectedParenAfter : SimpleDiagnostic<ExpectedParenAfter> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message = "Expected `(` after `{0}`.";

  auto Format() -> std::string {
    return llvm::formatv(Message, introducer.GetFixedSpelling()).str();
  }

  TokenKind introducer;
};

struct ExpectedCloseParen : SimpleDiagnostic<ExpectedCloseParen> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Unexpected tokens before `)`.";

  // TODO: Include the location of the matching open paren in the diagnostic.
  TokenizedBuffer::Token open_paren;
};

struct ExpectedSemiAfterExpression
    : SimpleDiagnostic<ExpectedSemiAfterExpression> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected `;` after expression.";
};

struct ExpectedSemiAfter : SimpleDiagnostic<ExpectedSemiAfter> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message = "Expected `;` after `{0}`.";

  auto Format() -> std::string {
    return llvm::formatv(Message, preceding.GetFixedSpelling()).str();
  }

  TokenKind preceding;
};

struct ExpectedIdentifierAfterDot
    : SimpleDiagnostic<ExpectedIdentifierAfterDot> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Expected identifier after `.`.";
};

struct UnexpectedTokenAfterListElement
    : SimpleDiagnostic<UnexpectedTokenAfterListElement> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message = "Expected `,` or `{0}`.";

  auto Format() -> std::string {
    return llvm::formatv(Message, close.GetFixedSpelling()).str();
  }

  TokenKind close;
};

struct BinaryOperatorRequiresWhitespace
    : SimpleDiagnostic<BinaryOperatorRequiresWhitespace> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message =
      "Whitespace missing {0} binary operator.";

  auto Format() -> std::string {
    const char* position = "around";
    if (has_leading_space) {
      position = "after";
    } else if (has_trailing_space) {
      position = "before";
    }
    return llvm::formatv(Message, position);
  }

  bool has_leading_space;
  bool has_trailing_space;
};

struct UnaryOperatorHasWhitespace
    : SimpleDiagnostic<UnaryOperatorHasWhitespace> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message =
      "Whitespace is not allowed {0} this unary operator.";

  auto Format() -> std::string {
    return llvm::formatv(Message, prefix ? "after" : "before");
  }

  bool prefix;
};

struct UnaryOperatorRequiresWhitespace
    : SimpleDiagnostic<UnaryOperatorRequiresWhitespace> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr const char* Message =
      "Whitespace is required {0} this unary operator.";

  auto Format() -> std::string {
    return llvm::formatv(Message, prefix ? "before" : "after");
  }

  bool prefix;
};

struct OperatorRequiresParentheses
    : SimpleDiagnostic<OperatorRequiresParentheses> {
  static constexpr llvm::StringLiteral ShortName = "syntax-error";
  static constexpr llvm::StringLiteral Message =
      "Parentheses are required to disambiguate operator precedence.";
};

ParseTree::Parser::Parser(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                          TokenDiagnosticEmitter& emitter)
    : tree_(tree_arg),
      tokens_(tokens_arg),
      emitter_(emitter),
      position_(tokens_.Tokens().begin()),
      end_(tokens_.Tokens().end()) {
  CHECK(std::find_if(position_, end_,
                     [&](TokenizedBuffer::Token t) {
                       return tokens_.GetKind(t) == TokenKind::EndOfFile();
                     }) != end_)
      << "No EndOfFileToken in token buffer.";
}

auto ParseTree::Parser::Parse(TokenizedBuffer& tokens,
                              TokenDiagnosticEmitter& emitter) -> ParseTree {
  ParseTree tree(tokens);

  // We expect to have a 1:1 correspondence between tokens and tree nodes, so
  // reserve the space we expect to need here to avoid allocation and copying
  // overhead.
  tree.node_impls_.reserve(tokens.Size());

  Parser parser(tree, tokens, emitter);
  while (!parser.AtEndOfFile()) {
    if (!parser.ParseDeclaration()) {
      // We don't have an enclosing parse tree node to mark as erroneous, so
      // just mark the tree as a whole.
      tree.has_errors_ = true;
    }
  }

  parser.AddLeafNode(ParseNodeKind::FileEnd(), *parser.position_);

  CHECK(tree.Verify()) << "Parse tree built but does not verify!";
  return tree;
}

auto ParseTree::Parser::Consume(TokenKind kind) -> TokenizedBuffer::Token {
  CHECK(kind != TokenKind::EndOfFile()) << "Cannot consume the EOF token!";
  CHECK(NextTokenIs(kind)) << "The current token is the wrong kind!";
  TokenizedBuffer::Token t = *position_;
  ++position_;
  CHECK(position_ != end_)
      << "Reached end of tokens without finding EOF token.";
  return t;
}

auto ParseTree::Parser::ConsumeIf(TokenKind kind)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (!NextTokenIs(kind)) {
    return {};
  }
  return Consume(kind);
}

auto ParseTree::Parser::AddLeafNode(ParseNodeKind kind,
                                    TokenizedBuffer::Token token) -> Node {
  Node n(tree_.node_impls_.size());
  tree_.node_impls_.push_back(NodeImpl(kind, token, /*subtree_size_arg=*/1));
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
  tree_.node_impls_[n.index_].has_error = true;
  tree_.has_errors_ = true;
}

// A marker for the start of a node's subtree.
//
// This is used to track the size of the node's subtree. It can be used
// repeatedly if multiple subtrees start at the same position.
struct ParseTree::Parser::SubtreeStart {
  int tree_size;
};

auto ParseTree::Parser::GetSubtreeStartPosition() -> SubtreeStart {
  return {static_cast<int>(tree_.node_impls_.size())};
}

auto ParseTree::Parser::AddNode(ParseNodeKind n_kind, TokenizedBuffer::Token t,
                                SubtreeStart start, bool has_error) -> Node {
  // The size of the subtree is the change in size from when we started this
  // subtree to now, but including the node we're about to add.
  int tree_stop_size = static_cast<int>(tree_.node_impls_.size()) + 1;
  int subtree_size = tree_stop_size - start.tree_size;

  Node n(tree_.node_impls_.size());
  tree_.node_impls_.push_back(NodeImpl(n_kind, t, subtree_size));
  if (has_error) {
    MarkNodeError(n);
  }

  return n;
}

auto ParseTree::Parser::SkipMatchingGroup() -> bool {
  TokenizedBuffer::Token t = *position_;
  TokenKind t_kind = tokens_.GetKind(t);
  if (!t_kind.IsOpeningSymbol()) {
    return false;
  }

  SkipTo(tokens_.GetMatchedClosingToken(t));
  Consume(t_kind.GetClosingSymbol());
  return true;
}

auto ParseTree::Parser::SkipTo(TokenizedBuffer::Token t) -> void {
  CHECK(t >= *position_) << "Tried to skip backwards.";
  position_ = TokenizedBuffer::TokenIterator(t);
  CHECK(position_ != end_) << "Skipped past EOF.";
}

auto ParseTree::Parser::FindNextOf(
    std::initializer_list<TokenKind> desired_kinds)
    -> llvm::Optional<TokenizedBuffer::Token> {
  auto new_position = position_;
  while (true) {
    TokenizedBuffer::Token token = *new_position;
    TokenKind kind = tokens_.GetKind(token);
    if (kind.IsOneOf(desired_kinds)) {
      return token;
    }

    // Step to the next token at the current bracketing level.
    if (kind.IsClosingSymbol() || kind == TokenKind::EndOfFile()) {
      // There are no more tokens at this level.
      return llvm::None;
    } else if (kind.IsOpeningSymbol()) {
      new_position =
          TokenizedBuffer::TokenIterator(tokens_.GetMatchedClosingToken(token));
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
    if (NextTokenKind() == TokenKind::CloseCurlyBrace()) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return llvm::None;
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(TokenKind::Semi())) {
      return on_semi(*semi);
    }

    // Skip over any matching group of tokens_.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    Consume(NextTokenKind());
  } while (!AtEndOfFile() &&
           is_same_line_or_indent_greater_than_root(*position_));

  return llvm::None;
}

auto ParseTree::Parser::ParseCloseParen(TokenizedBuffer::Token open_paren,
                                        ParseNodeKind kind)
    -> llvm::Optional<Node> {
  if (auto close_paren =
          ConsumeAndAddLeafNodeIf(TokenKind::CloseParen(), kind)) {
    return close_paren;
  }

  emitter_.EmitError<ExpectedCloseParen>(*position_,
                                         {.open_paren = open_paren});
  SkipTo(tokens_.GetMatchedClosingToken(open_paren));
  AddLeafNode(kind, Consume(TokenKind::CloseParen()));
  return llvm::None;
}

template <typename ListElementParser, typename ListCompletionHandler>
auto ParseTree::Parser::ParseList(TokenKind open, TokenKind close,
                                  ListElementParser list_element_parser,
                                  ParseNodeKind comma_kind,
                                  ListCompletionHandler list_handler,
                                  bool allow_trailing_comma)
    -> llvm::Optional<Node> {
  // `(` element-list[opt] `)`
  //
  // element-list ::= element
  //              ::= element `,` element-list
  TokenizedBuffer::Token open_paren = Consume(open);

  bool has_errors = false;
  bool any_commas = false;
  int64_t num_elements = 0;

  // Parse elements, if any are specified.
  if (!NextTokenIs(close)) {
    while (true) {
      bool element_error = !list_element_parser();
      has_errors |= element_error;
      ++num_elements;

      if (!NextTokenIsOneOf({close, TokenKind::Comma()})) {
        if (!element_error) {
          emitter_.EmitError<UnexpectedTokenAfterListElement>(*position_,
                                                              {.close = close});
        }
        has_errors = true;

        auto end_of_element = FindNextOf({TokenKind::Comma(), close});
        // The lexer guarantees that parentheses are balanced.
        CHECK(end_of_element) << "missing matching `)` for `(`";
        SkipTo(*end_of_element);
      }

      if (NextTokenIs(close)) {
        break;
      }

      AddLeafNode(comma_kind, Consume(TokenKind::Comma()));
      any_commas = true;

      if (allow_trailing_comma && NextTokenIs(close)) {
        break;
      }
    }
  }

  bool is_single_item = num_elements == 1 && !any_commas;
  return list_handler(open_paren, is_single_item, Consume(close), has_errors);
}

auto ParseTree::Parser::ParsePattern(PatternKind kind) -> llvm::Optional<Node> {
  if (NextTokenIs(TokenKind::Identifier()) &&
      tokens_.GetKind(*(position_ + 1)) == TokenKind::Colon()) {
    // identifier `:` type
    auto start = GetSubtreeStartPosition();
    AddLeafNode(ParseNodeKind::DeclaredName(),
                Consume(TokenKind::Identifier()));
    auto colon = Consume(TokenKind::Colon());
    auto type = ParseType();
    return AddNode(ParseNodeKind::PatternBinding(), colon, start,
                   /*has_error=*/!type);
  }

  switch (kind) {
    case PatternKind::Parameter:
      emitter_.EmitError<ExpectedParameterName>(*position_);
      break;

    case PatternKind::Variable:
      emitter_.EmitError<ExpectedVariableName>(*position_);
      break;
  }

  return llvm::None;
}

auto ParseTree::Parser::ParseFunctionParameter() -> llvm::Optional<Node> {
  return ParsePattern(PatternKind::Parameter);
}

auto ParseTree::Parser::ParseFunctionSignature() -> bool {
  auto start = GetSubtreeStartPosition();

  auto params = ParseParenList(
      [&] { return ParseFunctionParameter(); },
      ParseNodeKind::ParameterListComma(),
      [&](TokenizedBuffer::Token open_paren, bool /*is_single_item*/,
          TokenizedBuffer::Token close_paren, bool has_errors) {
        AddLeafNode(ParseNodeKind::ParameterListEnd(), close_paren);
        return AddNode(ParseNodeKind::ParameterList(), open_paren, start,
                       has_errors);
      });

  auto start_return_type = GetSubtreeStartPosition();
  if (auto arrow = ConsumeIf(TokenKind::MinusGreater())) {
    auto return_type = ParseType();
    AddNode(ParseNodeKind::ReturnType(), *arrow, start_return_type,
            /*has_error=*/!return_type);
    if (!return_type) {
      return false;
    }
  }

  return params.hasValue();
}

auto ParseTree::Parser::ParseCodeBlock() -> llvm::Optional<Node> {
  llvm::Optional<TokenizedBuffer::Token> maybe_open_curly =
      ConsumeIf(TokenKind::OpenCurlyBrace());
  if (!maybe_open_curly) {
    // Recover by parsing a single statement.
    emitter_.EmitError<ExpectedCodeBlock>(*position_);
    return ParseStatement();
  }
  TokenizedBuffer::Token open_curly = *maybe_open_curly;

  auto start = GetSubtreeStartPosition();

  bool has_errors = false;

  // Loop over all the different possibly nested elements in the code block.
  while (!NextTokenIs(TokenKind::CloseCurlyBrace())) {
    if (!ParseStatement()) {
      // We detected and diagnosed an error of some kind. We can trivially skip
      // to the actual close curly brace from here.
      // FIXME: It would be better to skip to the next semicolon, or the next
      // token at the start of a line with the same indent as this one.
      SkipTo(tokens_.GetMatchedClosingToken(open_curly));
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
  auto start = GetSubtreeStartPosition();

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
    emitter_.EmitError<ExpectedFunctionName>(*position_);
    // FIXME: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    return add_error_function_node();
  }

  TokenizedBuffer::Token open_paren = *position_;
  if (tokens_.GetKind(open_paren) != TokenKind::OpenParen()) {
    emitter_.EmitError<ExpectedFunctionParams>(open_paren);
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    return add_error_function_node();
  }
  TokenizedBuffer::Token close_paren =
      tokens_.GetMatchedClosingToken(open_paren);

  if (!ParseFunctionSignature()) {
    // Don't try to parse more of the function declaration, but consume a
    // declaration ending semicolon if found (without going to a new line).
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    return add_error_function_node();
  }

  // See if we should parse a definition which is represented as a code block.
  if (NextTokenIs(TokenKind::OpenCurlyBrace())) {
    if (!ParseCodeBlock()) {
      return add_error_function_node();
    }
  } else if (!ConsumeAndAddLeafNodeIf(TokenKind::Semi(),
                                      ParseNodeKind::DeclarationEnd())) {
    emitter_.EmitError<ExpectedFunctionBodyOrSemi>(*position_);
    if (tokens_.GetLine(*position_) == tokens_.GetLine(close_paren)) {
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
  // `var` pattern [= expression] `;`
  TokenizedBuffer::Token var_token = Consume(TokenKind::VarKeyword());
  auto start = GetSubtreeStartPosition();

  auto pattern = ParsePattern(PatternKind::Variable);
  if (!pattern) {
    if (auto after_pattern =
            FindNextOf({TokenKind::Equal(), TokenKind::Semi()})) {
      SkipTo(*after_pattern);
    }
  }

  auto start_init = GetSubtreeStartPosition();
  if (auto equal_token = ConsumeIf(TokenKind::Equal())) {
    auto init = ParseExpression();
    AddNode(ParseNodeKind::VariableInitializer(), *equal_token, start_init,
            /*has_error=*/!init);
  }

  auto semi = ConsumeAndAddLeafNodeIf(TokenKind::Semi(),
                                      ParseNodeKind::DeclarationEnd());
  if (!semi) {
    emitter_.EmitError<ExpectedSemiAfterExpression>(*position_);
    SkipPastLikelyEnd(var_token, [&](TokenizedBuffer::Token semi) {
      return AddLeafNode(ParseNodeKind::DeclarationEnd(), semi);
    });
  }

  return AddNode(ParseNodeKind::VariableDeclaration(), var_token, start,
                 /*has_error=*/!pattern || !semi);
}

auto ParseTree::Parser::ParseEmptyDeclaration() -> Node {
  return AddLeafNode(ParseNodeKind::EmptyDeclaration(),
                     Consume(TokenKind::Semi()));
}

auto ParseTree::Parser::ParseDeclaration() -> llvm::Optional<Node> {
  switch (NextTokenKind()) {
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
  emitter_.EmitError<UnrecognizedDeclaration>(*position_);

  // Skip forward past any end of a declaration we simply didn't understand so
  // that we can find the start of the next declaration or the end of a scope.
  if (auto found_semi_n =
          SkipPastLikelyEnd(*position_, [&](TokenizedBuffer::Token semi) {
            return AddLeafNode(ParseNodeKind::EmptyDeclaration(), semi);
          })) {
    MarkNodeError(*found_semi_n);
    return *found_semi_n;
  }

  // Nothing, not even a semicolon found.
  return llvm::None;
}

auto ParseTree::Parser::ParseParenExpression() -> llvm::Optional<Node> {
  // parenthesized-expression ::= `(` expression `)`
  // tuple-literal ::= `(` `)`
  //               ::= `(` expression `,` [expression-list [`,`]] `)`
  //
  // Parse the union of these, `(` [expression-list [`,`]] `)`, and work out
  // whether it's a tuple or a parenthesized expression afterwards.
  auto start = GetSubtreeStartPosition();
  return ParseParenList(
      [&] { return ParseExpression(); }, ParseNodeKind::TupleLiteralComma(),
      [&](TokenizedBuffer::Token open_paren, bool is_single_item,
          TokenizedBuffer::Token close_paren, bool has_arg_errors) {
        AddLeafNode(is_single_item ? ParseNodeKind::ParenExpressionEnd()
                                   : ParseNodeKind::TupleLiteralEnd(),
                    close_paren);
        return AddNode(is_single_item ? ParseNodeKind::ParenExpression()
                                      : ParseNodeKind::TupleLiteral(),
                       open_paren, start, has_arg_errors);
      },
      /*allow_trailing_comma=*/true);
}

auto ParseTree::Parser::ParseBraceExpression() -> llvm::Optional<Node> {
  // braced-expression ::= `{` [field-value-list] `}`
  //                   ::= `{` field-type-list `}`
  // field-value-list ::= field-value [`,`]
  //                  ::= field-value `,` field-value-list
  // field-value ::= `.` identifier `=` expression
  // field-type-list ::= field-type [`,`]
  //                 ::= field-type `,` field-type-list
  // field-type ::= `.` identifier `:` type
  //
  // Note that `{` `}` is the first form (an empty struct), but that an empty
  // struct value also behaves as an empty struct type.
  auto start = GetSubtreeStartPosition();
  enum Kind { Unknown, Value, Type };
  Kind kind = Unknown;
  return ParseList(
      TokenKind::OpenCurlyBrace(), TokenKind::CloseCurlyBrace(),
      [&]() -> llvm::Optional<Node> {
        auto start_elem = GetSubtreeStartPosition();

        auto diagnose_invalid_syntax = [&] {
          emitter_.EmitError<ExpectedStructLiteralField>(
              *position_,
              {.can_be_type = kind != Value, .can_be_value = kind != Type});
          return llvm::None;
        };

        if (!NextTokenIs(TokenKind::Period())) {
          return diagnose_invalid_syntax();
        }
        auto designator = ParseDesignatorExpression(
            start_elem, ParseNodeKind::StructFieldDesignator(),
            /*has_errors=*/false);
        if (!designator) {
          auto recovery_pos = FindNextOf(
              {TokenKind::Equal(), TokenKind::Colon(), TokenKind::Comma()});
          if (!recovery_pos ||
              tokens_.GetKind(*recovery_pos) == TokenKind::Comma()) {
            return llvm::None;
          }
          SkipTo(*recovery_pos);
        }

        // Work out the kind of this element
        Kind elem_kind =
            (NextTokenIs(TokenKind::Equal())
                 ? Value
                 : NextTokenIs(TokenKind::Colon()) ? Type : Unknown);
        if (elem_kind == Unknown || (kind != Unknown && elem_kind != kind)) {
          return diagnose_invalid_syntax();
        }
        kind = elem_kind;

        // Struct type fields and value fields use the same grammar except that
        // one has a `:` separator and the other has an `=` separator.
        auto equal_or_colon_token =
            Consume(kind == Type ? TokenKind::Colon() : TokenKind::Equal());
        auto type_or_value = ParseExpression();
        return AddNode(kind == Type ? ParseNodeKind::StructFieldType()
                                    : ParseNodeKind::StructFieldValue(),
                       equal_or_colon_token, start_elem,
                       /*has_error=*/!designator || !type_or_value);
      },
      ParseNodeKind::StructComma(),
      [&](TokenizedBuffer::Token open_brace, bool /*is_single_item*/,
          TokenizedBuffer::Token close_brace, bool has_errors) {
        AddLeafNode(ParseNodeKind::StructEnd(), close_brace);
        return AddNode(kind == Type ? ParseNodeKind::StructTypeLiteral()
                                    : ParseNodeKind::StructLiteral(),
                       open_brace, start, has_errors);
      },
      /*allow_trailing_comma=*/true);
}

auto ParseTree::Parser::ParsePrimaryExpression() -> llvm::Optional<Node> {
  llvm::Optional<ParseNodeKind> kind;
  switch (NextTokenKind()) {
    case TokenKind::Identifier():
      kind = ParseNodeKind::NameReference();
      break;

    case TokenKind::IntegerLiteral():
    case TokenKind::RealLiteral():
    case TokenKind::StringLiteral():
    case TokenKind::IntegerTypeLiteral():
    case TokenKind::UnsignedIntegerTypeLiteral():
    case TokenKind::FloatingPointTypeLiteral():
      kind = ParseNodeKind::Literal();
      break;

    case TokenKind::OpenParen():
      return ParseParenExpression();

    case TokenKind::OpenCurlyBrace():
      return ParseBraceExpression();

    default:
      emitter_.EmitError<ExpectedExpression>(*position_);
      return llvm::None;
  }

  return AddLeafNode(*kind, Consume(NextTokenKind()));
}

auto ParseTree::Parser::ParseDesignatorExpression(SubtreeStart start,
                                                  ParseNodeKind kind,
                                                  bool has_errors)
    -> llvm::Optional<Node> {
  // `.` identifier
  auto dot = Consume(TokenKind::Period());
  auto name = ConsumeIf(TokenKind::Identifier());
  if (name) {
    AddLeafNode(ParseNodeKind::DesignatedName(), *name);
  } else {
    emitter_.EmitError<ExpectedIdentifierAfterDot>(*position_);
    // If we see a keyword, assume it was intended to be the designated name.
    // TODO: Should keywords be valid in designators?
    if (NextTokenKind().IsKeyword()) {
      name = Consume(NextTokenKind());
      auto name_node = AddLeafNode(ParseNodeKind::DesignatedName(), *name);
      MarkNodeError(name_node);
    } else {
      has_errors = true;
    }
  }

  Node result = AddNode(kind, dot, start, has_errors);
  return name ? result : llvm::Optional<Node>();
}

auto ParseTree::Parser::ParseCallExpression(SubtreeStart start, bool has_errors)
    -> llvm::Optional<Node> {
  // `(` expression-list[opt] `)`
  //
  // expression-list ::= expression
  //                 ::= expression `,` expression-list
  return ParseParenList(
      [&] { return ParseExpression(); }, ParseNodeKind::CallExpressionComma(),
      [&](TokenizedBuffer::Token open_paren, bool /*is_single_item*/,
          TokenizedBuffer::Token close_paren, bool has_arg_errors) {
        AddLeafNode(ParseNodeKind::CallExpressionEnd(), close_paren);
        return AddNode(ParseNodeKind::CallExpression(), open_paren, start,
                       has_errors || has_arg_errors);
      });
}

auto ParseTree::Parser::ParsePostfixExpression() -> llvm::Optional<Node> {
  auto start = GetSubtreeStartPosition();
  llvm::Optional<Node> expression = ParsePrimaryExpression();

  while (true) {
    switch (NextTokenKind()) {
      case TokenKind::Period():
        expression = ParseDesignatorExpression(
            start, ParseNodeKind::DesignatorExpression(), !expression);
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

// Determines whether the given token is considered to be the start of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedStartOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::OpenParen(), TokenKind::Identifier(),
                       TokenKind::IntegerLiteral(), TokenKind::RealLiteral(),
                       TokenKind::StringLiteral()});
}

// Determines whether the given token is considered to be the end of an operand
// according to the rules for infix operator parsing.
static auto IsAssumedEndOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::CloseParen(), TokenKind::CloseCurlyBrace(),
                       TokenKind::CloseSquareBracket(), TokenKind::Identifier(),
                       TokenKind::IntegerLiteral(), TokenKind::RealLiteral(),
                       TokenKind::StringLiteral()});
}

// Determines whether the given token could possibly be the start of an operand.
// This is conservatively correct, and will never incorrectly return `false`,
// but can incorrectly return `true`.
static auto IsPossibleStartOfOperand(TokenKind kind) -> bool {
  return !kind.IsOneOf({TokenKind::CloseParen(), TokenKind::CloseCurlyBrace(),
                        TokenKind::CloseSquareBracket(), TokenKind::Comma(),
                        TokenKind::Semi(), TokenKind::Colon()});
}

auto ParseTree::Parser::IsLexicallyValidInfixOperator() -> bool {
  CHECK(!AtEndOfFile()) << "Expected an operator token.";

  bool leading_space = tokens_.HasLeadingWhitespace(*position_);
  bool trailing_space = tokens_.HasTrailingWhitespace(*position_);

  // If there's whitespace on both sides, it's an infix operator.
  if (leading_space && trailing_space) {
    return true;
  }

  // If there's whitespace on exactly one side, it's not an infix operator.
  if (leading_space || trailing_space) {
    return false;
  }

  // Otherwise, for an infix operator, the preceding token must be any close
  // bracket, identifier, or literal and the next token must be an open paren,
  // identifier, or literal.
  if (position_ == tokens_.Tokens().begin() ||
      !IsAssumedEndOfOperand(tokens_.GetKind(*(position_ - 1))) ||
      !IsAssumedStartOfOperand(tokens_.GetKind(*(position_ + 1)))) {
    return false;
  }

  return true;
}

auto ParseTree::Parser::DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
  bool is_valid_as_infix = IsLexicallyValidInfixOperator();

  if (fixity == OperatorFixity::Infix) {
    // Infix operators must satisfy the infix operator rules.
    if (!is_valid_as_infix) {
      emitter_.EmitError<BinaryOperatorRequiresWhitespace>(
          *position_,
          {.has_leading_space = tokens_.HasLeadingWhitespace(*position_),
           .has_trailing_space = tokens_.HasTrailingWhitespace(*position_)});
    }
  } else {
    bool prefix = fixity == OperatorFixity::Prefix;

    // Whitespace is not permitted between a symbolic pre/postfix operator and
    // its operand.
    if (NextTokenKind().IsSymbol() &&
        (prefix ? tokens_.HasTrailingWhitespace(*position_)
                : tokens_.HasLeadingWhitespace(*position_))) {
      emitter_.EmitError<UnaryOperatorHasWhitespace>(*position_,
                                                     {.prefix = prefix});
    }
    // Pre/postfix operators must not satisfy the infix operator rules.
    if (is_valid_as_infix) {
      emitter_.EmitError<UnaryOperatorRequiresWhitespace>(*position_,
                                                          {.prefix = prefix});
    }
  }
}

auto ParseTree::Parser::IsTrailingOperatorInfix() -> bool {
  if (AtEndOfFile()) {
    return false;
  }

  // An operator that follows the infix operator rules is parsed as
  // infix, unless the next token means that it can't possibly be.
  if (IsLexicallyValidInfixOperator() &&
      IsPossibleStartOfOperand(tokens_.GetKind(*(position_ + 1)))) {
    return true;
  }

  // A trailing operator with leading whitespace that's not valid as infix is
  // not valid at all. If the next token looks like the start of an operand,
  // then parse as infix, otherwise as postfix. Either way we'll produce a
  // diagnostic later on.
  if (tokens_.HasLeadingWhitespace(*position_) &&
      IsAssumedStartOfOperand(tokens_.GetKind(*(position_ + 1)))) {
    return true;
  }

  return false;
}

auto ParseTree::Parser::ParseOperatorExpression(
    PrecedenceGroup ambient_precedence) -> llvm::Optional<Node> {
  auto start = GetSubtreeStartPosition();

  llvm::Optional<Node> lhs;
  PrecedenceGroup lhs_precedence = PrecedenceGroup::ForPostfixExpression();

  // Check for a prefix operator.
  if (auto operator_precedence = PrecedenceGroup::ForLeading(NextTokenKind());
      !operator_precedence) {
    lhs = ParsePostfixExpression();
  } else {
    if (PrecedenceGroup::GetPriority(ambient_precedence,
                                     *operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this prefix operator in this
      // context. Diagnose this, but carry on and parse it anyway.
      emitter_.EmitError<OperatorRequiresParentheses>(*position_);
    } else {
      // Check that this operator follows the proper whitespace rules.
      DiagnoseOperatorFixity(OperatorFixity::Prefix);
    }

    auto operator_token = Consume(NextTokenKind());
    bool has_errors = !ParseOperatorExpression(*operator_precedence);
    lhs = AddNode(ParseNodeKind::PrefixOperator(), operator_token, start,
                  has_errors);
    lhs_precedence = *operator_precedence;
  }

  // Consume a sequence of infix and postfix operators.
  while (auto trailing_operator = PrecedenceGroup::ForTrailing(
             NextTokenKind(), IsTrailingOperatorInfix())) {
    auto [operator_precedence, is_binary] = *trailing_operator;

    // FIXME: If this operator is ambiguous with either the ambient precedence
    // or the LHS precedence, and there's a variant with a different fixity
    // that would work, use that one instead for error recovery.
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
      emitter_.EmitError<OperatorRequiresParentheses>(*position_);
      lhs = llvm::None;
    } else {
      DiagnoseOperatorFixity(is_binary ? OperatorFixity::Infix
                                       : OperatorFixity::Postfix);
    }

    auto operator_token = Consume(NextTokenKind());

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

auto ParseTree::Parser::ParseType() -> llvm::Optional<Node> {
  return ParseOperatorExpression(PrecedenceGroup::ForType());
}

auto ParseTree::Parser::ParseExpressionStatement() -> llvm::Optional<Node> {
  TokenizedBuffer::Token start_token = *position_;
  auto start = GetSubtreeStartPosition();

  bool has_errors = !ParseExpression();

  if (auto semi = ConsumeIf(TokenKind::Semi())) {
    return AddNode(ParseNodeKind::ExpressionStatement(), *semi, start,
                   has_errors);
  }

  if (!has_errors) {
    emitter_.EmitError<ExpectedSemiAfterExpression>(*position_);
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

auto ParseTree::Parser::ParseParenCondition(TokenKind introducer)
    -> llvm::Optional<Node> {
  // `(` expression `)`
  auto start = GetSubtreeStartPosition();
  auto open_paren = ConsumeIf(TokenKind::OpenParen());
  if (!open_paren) {
    emitter_.EmitError<ExpectedParenAfter>(*position_,
                                           {.introducer = introducer});
  }

  auto expr = ParseExpression();

  if (!open_paren) {
    // Don't expect a matching closing paren if there wasn't an opening paren.
    return llvm::None;
  }

  auto close_paren =
      ParseCloseParen(*open_paren, ParseNodeKind::ConditionEnd());

  return AddNode(ParseNodeKind::Condition(), *open_paren, start,
                 /*has_error=*/!expr || !close_paren);
}

auto ParseTree::Parser::ParseIfStatement() -> llvm::Optional<Node> {
  auto start = GetSubtreeStartPosition();
  auto if_token = Consume(TokenKind::IfKeyword());
  auto cond = ParseParenCondition(TokenKind::IfKeyword());
  auto then_case = ParseCodeBlock();
  bool else_has_errors = false;
  if (ConsumeAndAddLeafNodeIf(TokenKind::ElseKeyword(),
                              ParseNodeKind::IfStatementElse())) {
    // 'else if' is permitted as a special case.
    if (NextTokenIs(TokenKind::IfKeyword())) {
      else_has_errors = !ParseIfStatement();
    } else {
      else_has_errors = !ParseCodeBlock();
    }
  }
  return AddNode(ParseNodeKind::IfStatement(), if_token, start,
                 /*has_error=*/!cond || !then_case || else_has_errors);
}

auto ParseTree::Parser::ParseWhileStatement() -> llvm::Optional<Node> {
  auto start = GetSubtreeStartPosition();
  auto while_token = Consume(TokenKind::WhileKeyword());
  auto cond = ParseParenCondition(TokenKind::WhileKeyword());
  auto body = ParseCodeBlock();
  return AddNode(ParseNodeKind::WhileStatement(), while_token, start,
                 /*has_error=*/!cond || !body);
}

auto ParseTree::Parser::ParseKeywordStatement(ParseNodeKind kind,
                                              KeywordStatementArgument argument)
    -> llvm::Optional<Node> {
  auto keyword_kind = NextTokenKind();
  assert(keyword_kind.IsKeyword());

  auto start = GetSubtreeStartPosition();
  auto keyword = Consume(keyword_kind);

  bool arg_error = false;
  if ((argument == KeywordStatementArgument::Optional &&
       NextTokenKind() != TokenKind::Semi()) ||
      argument == KeywordStatementArgument::Mandatory) {
    arg_error = !ParseExpression();
  }

  auto semi =
      ConsumeAndAddLeafNodeIf(TokenKind::Semi(), ParseNodeKind::StatementEnd());
  if (!semi) {
    emitter_.EmitError<ExpectedSemiAfter>(*position_,
                                          {.preceding = keyword_kind});
    // FIXME: Try to skip to a semicolon to recover.
  }
  return AddNode(kind, keyword, start, /*has_error=*/!semi || arg_error);
}

auto ParseTree::Parser::ParseStatement() -> llvm::Optional<Node> {
  switch (NextTokenKind()) {
    case TokenKind::VarKeyword():
      return ParseVariableDeclaration();

    case TokenKind::IfKeyword():
      return ParseIfStatement();

    case TokenKind::WhileKeyword():
      return ParseWhileStatement();

    case TokenKind::ContinueKeyword():
      return ParseKeywordStatement(ParseNodeKind::ContinueStatement(),
                                   KeywordStatementArgument::None);

    case TokenKind::BreakKeyword():
      return ParseKeywordStatement(ParseNodeKind::BreakStatement(),
                                   KeywordStatementArgument::None);

    case TokenKind::ReturnKeyword():
      return ParseKeywordStatement(ParseNodeKind::ReturnStatement(),
                                   KeywordStatementArgument::Optional);

    default:
      // A statement with no introducer token can only be an expression
      // statement.
      return ParseExpressionStatement();
  }
}

}  // namespace Carbon

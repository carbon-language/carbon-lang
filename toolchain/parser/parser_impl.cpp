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

CARBON_DIAGNOSTIC(ExpectedSemiAfterExpression, Error,
                  "Expected `;` after expression.");

// Manages the parser's stack depth, particularly decrementing on destruction.
// This should only be instantiated through RETURN_IF_STACK_LIMITED.
class ParseTree::Parser::ScopedStackStep {
 public:
  explicit ScopedStackStep(ParseTree::Parser* parser) : parser_(parser) {
    ++parser_->stack_depth_;
  }
  ~ScopedStackStep() { --parser_->stack_depth_; }

  auto VerifyUnderLimit() -> bool {
    if (parser_->stack_depth_ >= StackDepthLimit) {
      CARBON_DIAGNOSTIC(StackLimitExceeded, Error,
                        "Exceeded recursion limit ({0})", int);
      parser_->emitter_.Emit(*parser_->position_, StackLimitExceeded,
                             ParseTree::StackDepthLimit);
      return false;
    }
    return true;
  }

 private:
  ParseTree::Parser* parser_;
};

// Encapsulates checking the stack and erroring if needed. This should be called
// at the start of every parse function.
#define CARBON_RETURN_IF_STACK_LIMITED(error_return_expr) \
  ScopedStackStep scoped_stack_step(this);                \
  if (!scoped_stack_step.VerifyUnderLimit()) {            \
    return (error_return_expr);                           \
  }

// A relative location for characters in errors.
enum class RelativeLocation : int8_t {
  Around,
  After,
  Before,
};

// Adapts RelativeLocation for use with formatv.
static auto operator<<(llvm::raw_ostream& out, RelativeLocation loc)
    -> llvm::raw_ostream& {
  switch (loc) {
    case RelativeLocation::Around:
      out << "around";
      break;
    case RelativeLocation::After:
      out << "after";
      break;
    case RelativeLocation::Before:
      out << "before";
      break;
  }
  return out;
}

ParseTree::Parser::Parser(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                          TokenDiagnosticEmitter& emitter)
    : tree_(tree_arg),
      tokens_(tokens_arg),
      emitter_(emitter),
      position_(tokens_.tokens().begin()),
      end_(tokens_.tokens().end()) {
  CARBON_CHECK(std::find_if(position_, end_,
                            [&](TokenizedBuffer::Token t) {
                              return tokens_.GetKind(t) ==
                                     TokenKind::EndOfFile();
                            }) != end_)
      << "No EndOfFileToken in token buffer.";
}

auto ParseTree::Parser::Parse(TokenizedBuffer& tokens,
                              TokenDiagnosticEmitter& emitter) -> ParseTree {
  ParseTree tree(tokens);

  // We expect to have a 1:1 correspondence between tokens and tree nodes, so
  // reserve the space we expect to need here to avoid allocation and copying
  // overhead.
  tree.node_impls_.reserve(tokens.size());

  Parser parser(tree, tokens, emitter);
  while (!parser.AtEndOfFile()) {
    if (!parser.ParseDeclaration()) {
      // We don't have an enclosing parse tree node to mark as erroneous, so
      // just mark the tree as a whole.
      tree.has_errors_ = true;
    }
  }

  parser.AddLeafNode(ParseNodeKind::FileEnd(), *parser.position_);

  CARBON_CHECK(tree.Verify()) << "Parse tree built but does not verify!";
  return tree;
}

auto ParseTree::Parser::Consume(TokenKind kind) -> TokenizedBuffer::Token {
  CARBON_CHECK(kind != TokenKind::EndOfFile())
      << "Cannot consume the EOF token!";
  CARBON_CHECK(NextTokenIs(kind)) << "The current token is the wrong kind!";
  TokenizedBuffer::Token t = *position_;
  ++position_;
  CARBON_CHECK(position_ != end_)
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
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards.";
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
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
      // Advance past the closing token.
      ++new_position;
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

  // TODO: Include the location of the matching open_paren in the diagnostic.
  CARBON_DIAGNOSTIC(ExpectedCloseParen, Error, "Unexpected tokens before `)`.");
  emitter_.Emit(*position_, ExpectedCloseParen);
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
          CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                            "Expected `,` or `{0}`.", TokenKind);
          emitter_.Emit(*position_, UnexpectedTokenAfterListElement, close);
        }
        has_errors = true;

        auto end_of_element = FindNextOf({TokenKind::Comma(), close});
        // The lexer guarantees that parentheses are balanced.
        CARBON_CHECK(end_of_element) << "missing matching `)` for `(`";
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
      CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                        "Expected parameter declaration.");
      emitter_.Emit(*position_, ExpectedParameterName);
      break;

    case PatternKind::Variable:
      CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                        "Expected pattern in `var` declaration.");
      emitter_.Emit(*position_, ExpectedVariableName);
      break;
  }

  return llvm::None;
}

auto ParseTree::Parser::ParseFunctionParameter() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  return ParsePattern(PatternKind::Parameter);
}

auto ParseTree::Parser::ParseFunctionSignature() -> bool {
  CARBON_RETURN_IF_STACK_LIMITED(false);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  llvm::Optional<TokenizedBuffer::Token> maybe_open_curly =
      ConsumeIf(TokenKind::OpenCurlyBrace());
  if (!maybe_open_curly) {
    // Recover by parsing a single statement.
    CARBON_DIAGNOSTIC(ExpectedCodeBlock, Error, "Expected braced code block.");
    emitter_.Emit(*position_, ExpectedCodeBlock);
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
      // TODO: It would be better to skip to the next semicolon, or the next
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

auto ParseTree::Parser::ParsePackageDirective() -> Node {
  TokenizedBuffer::Token package_intro_token = Consume(TokenKind::Package());
  auto package_start = GetSubtreeStartPosition();
  auto create_error_node = [&]() {
    return AddNode(ParseNodeKind::PackageDirective(), package_intro_token,
                   package_start,
                   /*has_error=*/true);
  };

  CARBON_RETURN_IF_STACK_LIMITED(create_error_node());

  auto exit_on_parse_error = [&]() {
    SkipPastLikelyEnd(package_intro_token, [&](TokenizedBuffer::Token semi) {
      return AddLeafNode(ParseNodeKind::PackageEnd(), semi);
    });

    return create_error_node();
  };

  if (!NextTokenIs(TokenKind::Identifier())) {
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterPackage, Error,
                      "Expected identifier after `package`.");
    emitter_.Emit(*position_, ExpectedIdentifierAfterPackage);
    return exit_on_parse_error();
  }

  AddLeafNode(ParseNodeKind::DeclaredName(), Consume(TokenKind::Identifier()));
  bool library_parsed = false;

  if (tokens_.GetKind(*(position_)) == TokenKind::Library()) {
    auto library_start = GetSubtreeStartPosition();
    auto library_decl_token = Consume(TokenKind::Library());

    if (tokens_.GetKind(*(position_)) != TokenKind::StringLiteral()) {
      CARBON_DIAGNOSTIC(
          ExpectedLibraryName, Error,
          "Expected a string literal to specify the library name.");
      emitter_.Emit(*position_, ExpectedLibraryName);
      return exit_on_parse_error();
    }

    AddLeafNode(ParseNodeKind::Literal(), Consume(TokenKind::StringLiteral()));
    AddNode(ParseNodeKind::PackageLibrary(), library_decl_token, library_start,
            /*has_error=*/false);
    library_parsed = true;
  }

  auto api_or_impl_token = tokens_.GetKind(*(position_));

  if (api_or_impl_token == TokenKind::Api()) {
    AddLeafNode(ParseNodeKind::PackageApi(), Consume(TokenKind::Api()));
  } else if (api_or_impl_token == TokenKind::Impl()) {
    AddLeafNode(ParseNodeKind::PackageImpl(), Consume(TokenKind::Impl()));
  } else if (!library_parsed &&
             api_or_impl_token == TokenKind::StringLiteral()) {
    // If we come acroess a string literal and we didn't parse `library "..."`
    // yet, then most probably the user forgot to add `library` before the
    // library name.
    CARBON_DIAGNOSTIC(MissingLibraryKeyword, Error,
                      "Missing `library` keyword.");
    emitter_.Emit(*position_, MissingLibraryKeyword);
    return exit_on_parse_error();
  } else {
    CARBON_DIAGNOSTIC(ExpectedApiOrImpl, Error, "Expected a `api` or `impl`.");
    emitter_.Emit(*position_, ExpectedApiOrImpl);
    return exit_on_parse_error();
  }

  if (tokens_.GetKind(*(position_)) != TokenKind::Semi()) {
    CARBON_DIAGNOSTIC(ExpectedSemiToEndPackageDirective, Error,
                      "Expected `;` to end package directive.");
    emitter_.Emit(*position_, ExpectedSemiToEndPackageDirective);
    return exit_on_parse_error();
  }

  AddLeafNode(ParseNodeKind::PackageEnd(), Consume(TokenKind::Semi()));

  return AddNode(ParseNodeKind::PackageDirective(), package_intro_token,
                 package_start, /*has_error=*/false);
}

auto ParseTree::Parser::ParseFunctionDeclaration() -> Node {
  TokenizedBuffer::Token function_intro_token = Consume(TokenKind::Fn());
  auto start = GetSubtreeStartPosition();

  auto add_error_function_node = [&] {
    return AddNode(ParseNodeKind::FunctionDeclaration(), function_intro_token,
                   start, /*has_error=*/true);
  };
  CARBON_RETURN_IF_STACK_LIMITED(add_error_function_node());

  auto handle_semi_in_error_recovery = [&](TokenizedBuffer::Token semi) {
    return AddLeafNode(ParseNodeKind::DeclarationEnd(), semi);
  };

  auto name_n = ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                                        ParseNodeKind::DeclaredName());
  if (!name_n) {
    CARBON_DIAGNOSTIC(ExpectedFunctionName, Error,
                      "Expected function name after `fn` keyword.");
    emitter_.Emit(*position_, ExpectedFunctionName);
    // TODO: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    SkipPastLikelyEnd(function_intro_token, handle_semi_in_error_recovery);
    return add_error_function_node();
  }

  TokenizedBuffer::Token open_paren = *position_;
  if (tokens_.GetKind(open_paren) != TokenKind::OpenParen()) {
    CARBON_DIAGNOSTIC(ExpectedFunctionParams, Error,
                      "Expected `(` after function name.");
    emitter_.Emit(open_paren, ExpectedFunctionParams);
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
    CARBON_DIAGNOSTIC(
        ExpectedFunctionBodyOrSemi, Error,
        "Expected function definition or `;` after function declaration.");
    emitter_.Emit(*position_, ExpectedFunctionBodyOrSemi);
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
  TokenizedBuffer::Token var_token = Consume(TokenKind::Var());
  auto start = GetSubtreeStartPosition();

  CARBON_RETURN_IF_STACK_LIMITED(AddNode(ParseNodeKind::VariableDeclaration(),
                                         var_token, start,
                                         /*has_error=*/true));

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
    emitter_.Emit(*position_, ExpectedSemiAfterExpression);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  switch (NextTokenKind()) {
    case TokenKind::Package():
      return ParsePackageDirective();
    case TokenKind::Fn():
      return ParseFunctionDeclaration();
    case TokenKind::Var():
      return ParseVariableDeclaration();
    case TokenKind::Semi():
      return ParseEmptyDeclaration();
    case TokenKind::EndOfFile():
      return llvm::None;
    default:
      // Errors are handled outside the switch.
      break;
  }

  // Should happen for packages now.
  // We didn't recognize an introducer for a valid declaration.
  CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                    "Unrecognized declaration introducer.");
  emitter_.Emit(*position_, UnrecognizedDeclaration);

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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
          CARBON_DIAGNOSTIC(ExpectedStructLiteralField, Error,
                            "Expected {0}{1}{2}.", llvm::StringRef,
                            llvm::StringRef, llvm::StringRef);
          bool can_be_type = kind != Value;
          bool can_be_value = kind != Type;
          emitter_.Emit(*position_, ExpectedStructLiteralField,
                        can_be_type ? "`.field: type`" : "",
                        (can_be_type && can_be_value) ? " or " : "",
                        can_be_value ? "`.field = value`" : "");
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
        Kind elem_kind = (NextTokenIs(TokenKind::Equal())   ? Value
                          : NextTokenIs(TokenKind::Colon()) ? Type
                                                            : Unknown);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      emitter_.Emit(*position_, ExpectedExpression);
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
    CARBON_DIAGNOSTIC(ExpectedIdentifierAfterDot, Error,
                      "Expected identifier after `.`.");
    emitter_.Emit(*position_, ExpectedIdentifierAfterDot);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  auto start = GetSubtreeStartPosition();
  llvm::Optional<Node> expression = ParsePrimaryExpression();

  TokenizedBuffer::TokenIterator last_position = position_;
  while (true) {
    switch (NextTokenKind()) {
      case TokenKind::Period():
        expression = ParseDesignatorExpression(
            start, ParseNodeKind::DesignatorExpression(), !expression);
        break;

      case TokenKind::OpenParen():
        expression = ParseCallExpression(start, !expression);
        break;

      default:
        return expression;
    }
    // This is subject to an infinite loop if a child call fails, so monitor for
    // stalling.
    if (last_position == position_) {
      CARBON_CHECK(expression == llvm::None);
      return expression;
    }
    last_position = position_;
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
  CARBON_CHECK(!AtEndOfFile()) << "Expected an operator token.";

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
  if (position_ == tokens_.tokens().begin() ||
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
      CARBON_DIAGNOSTIC(BinaryOperatorRequiresWhitespace, Error,
                        "Whitespace missing {0} binary operator.",
                        RelativeLocation);
      emitter_.Emit(*position_, BinaryOperatorRequiresWhitespace,
                    tokens_.HasLeadingWhitespace(*position_)
                        ? RelativeLocation::After
                        : (tokens_.HasTrailingWhitespace(*position_)
                               ? RelativeLocation::Before
                               : RelativeLocation::Around));
    }
  } else {
    bool prefix = fixity == OperatorFixity::Prefix;

    // Whitespace is not permitted between a symbolic pre/postfix operator and
    // its operand.
    if (NextTokenKind().IsSymbol() &&
        (prefix ? tokens_.HasTrailingWhitespace(*position_)
                : tokens_.HasLeadingWhitespace(*position_))) {
      CARBON_DIAGNOSTIC(UnaryOperatorHasWhitespace, Error,
                        "Whitespace is not allowed {0} this unary operator.",
                        RelativeLocation);
      emitter_.Emit(
          *position_, UnaryOperatorHasWhitespace,
          prefix ? RelativeLocation::After : RelativeLocation::Before);
    }
    // Pre/postfix operators must not satisfy the infix operator rules.
    if (is_valid_as_infix) {
      CARBON_DIAGNOSTIC(UnaryOperatorRequiresWhitespace, Error,
                        "Whitespace is required {0} this unary operator.",
                        RelativeLocation);
      emitter_.Emit(
          *position_, UnaryOperatorRequiresWhitespace,
          prefix ? RelativeLocation::Before : RelativeLocation::After);
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
  // May be omitted a couple different ways here.
  CARBON_DIAGNOSTIC(
      OperatorRequiresParentheses, Error,
      "Parentheses are required to disambiguate operator precedence.");

  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
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
      emitter_.Emit(*position_, OperatorRequiresParentheses);
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

    // TODO: If this operator is ambiguous with either the ambient precedence
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
      // LHS operator is a unary operator that can't be nested within
      // this operator. Either way, parentheses are required.
      emitter_.Emit(*position_, OperatorRequiresParentheses);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  return ParseOperatorExpression(PrecedenceGroup::ForTopLevelExpression());
}

auto ParseTree::Parser::ParseType() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  return ParseOperatorExpression(PrecedenceGroup::ForType());
}

auto ParseTree::Parser::ParseExpressionStatement() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  TokenizedBuffer::Token start_token = *position_;
  auto start = GetSubtreeStartPosition();

  bool has_errors = !ParseExpression();

  if (auto semi = ConsumeIf(TokenKind::Semi())) {
    return AddNode(ParseNodeKind::ExpressionStatement(), *semi, start,
                   has_errors);
  }

  if (!has_errors) {
    emitter_.Emit(*position_, ExpectedSemiAfterExpression);
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
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  // `(` expression `)`
  auto start = GetSubtreeStartPosition();
  auto open_paren = ConsumeIf(TokenKind::OpenParen());
  if (!open_paren) {
    CARBON_DIAGNOSTIC(ExpectedParenAfter, Error, "Expected `(` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedParenAfter, introducer);
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
  auto if_token = Consume(TokenKind::If());
  auto cond = ParseParenCondition(TokenKind::If());
  auto then_case = ParseCodeBlock();
  bool else_has_errors = false;
  if (ConsumeAndAddLeafNodeIf(TokenKind::Else(),
                              ParseNodeKind::IfStatementElse())) {
    // 'else if' is permitted as a special case.
    if (NextTokenIs(TokenKind::If())) {
      else_has_errors = !ParseIfStatement();
    } else {
      else_has_errors = !ParseCodeBlock();
    }
  }
  return AddNode(ParseNodeKind::IfStatement(), if_token, start,
                 /*has_error=*/!cond || !then_case || else_has_errors);
}

auto ParseTree::Parser::ParseWhileStatement() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  auto start = GetSubtreeStartPosition();
  auto while_token = Consume(TokenKind::While());
  auto cond = ParseParenCondition(TokenKind::While());
  auto body = ParseCodeBlock();
  return AddNode(ParseNodeKind::WhileStatement(), while_token, start,
                 /*has_error=*/!cond || !body);
}

auto ParseTree::Parser::ParseForStatement() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  auto for_start = GetSubtreeStartPosition();
  auto for_token = Consume(TokenKind::For());

  // Parse the loop header.
  auto header = [this]() -> llvm::Optional<Node> {
    auto open_paren = ConsumeIf(TokenKind::OpenParen());
    auto header_start = GetSubtreeStartPosition();

    if (!open_paren) {
      CARBON_DIAGNOSTIC(ExpectedParenAfter, Error,
                        "Expected `(` after `{0}`. Recovering from missing `(` "
                        "not implemented yet!",
                        TokenKind);
      emitter_.Emit(*position_, ExpectedParenAfter, TokenKind::For());
      // TODO: A proper recovery strategy is needed here. For now, I assume that
      // all brackets are properly balanced (i.e. each open bracket has a
      // closing one).
      // This is temporary until we come to a conclusion regarding the recovery
      // tokens strategy.
      return llvm::None;
    }

    bool iter_var_parsed = false;

    if (NextTokenIs(TokenKind::Var())) {
      auto var_token = Consume(TokenKind::Var());
      auto var_start = GetSubtreeStartPosition();
      auto pattern = ParsePattern(PatternKind::Variable);
      AddNode(ParseNodeKind::VariableDeclaration(), var_token, var_start,
              !pattern);
      iter_var_parsed = true;
    } else {
      CARBON_DIAGNOSTIC(ExpectedVariableDeclaration, Error,
                        "Expected `var` declaration.");
      emitter_.Emit(*position_, ExpectedVariableDeclaration);

      if (auto next_in = FindNextOf({TokenKind::In()}); next_in) {
        SkipTo(*next_in);
      }
    }

    // A separator is either an `in` or a `:`. Even though `:` is incorrect,
    // accidentally typing it by a C++ programmer might be a common mistake that
    // warrants special handling.
    bool separator_parsed = false;
    bool in_parsed = false;

    if (NextTokenIs(TokenKind::In())) {
      separator_parsed = true;
      in_parsed = true;
      AddLeafNode(ParseNodeKind::ForIn(), Consume(TokenKind::In()));
    } else if (NextTokenIs(TokenKind::Colon())) {
      separator_parsed = true;
      CARBON_DIAGNOSTIC(ExpectedIn, Error, "`:` should be replaced by `in`.");
      emitter_.Emit(*position_, ExpectedIn);
      Consume(TokenKind::Colon());
    } else {
      CARBON_DIAGNOSTIC(ExpectedIn, Error,
                        "Expected `in` after loop `var` declaration.");
      emitter_.Emit(*position_, ExpectedIn);
      SkipTo(tokens_.GetMatchedClosingToken(*open_paren));
    }

    // Only try to parse the container expression if a separator was parsed.
    // This reduces the emitted error messages if the separator was missing
    // altogether.
    auto container_expr = separator_parsed ? ParseExpression() : llvm::None;

    auto close_paren =
        ParseCloseParen(*open_paren, ParseNodeKind::ForHeaderEnd());

    return AddNode(
        ParseNodeKind::ForHeader(), *open_paren, header_start,
        !iter_var_parsed || !in_parsed || !container_expr || !close_paren);
  }();

  auto body = ParseCodeBlock();

  return AddNode(ParseNodeKind::ForStatement(), for_token, for_start,
                 !header || !body);
}

auto ParseTree::Parser::ParseKeywordStatement(ParseNodeKind kind,
                                              KeywordStatementArgument argument)
    -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  auto keyword_kind = NextTokenKind();
  CARBON_CHECK(keyword_kind.IsKeyword());

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
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedSemiAfter, keyword_kind);
    // TODO: Try to skip to a semicolon to recover.
  }
  return AddNode(kind, keyword, start, /*has_error=*/!semi || arg_error);
}

auto ParseTree::Parser::ParseStatement() -> llvm::Optional<Node> {
  CARBON_RETURN_IF_STACK_LIMITED(llvm::None);
  switch (NextTokenKind()) {
    case TokenKind::Var():
      return ParseVariableDeclaration();

    case TokenKind::If():
      return ParseIfStatement();

    case TokenKind::While():
      return ParseWhileStatement();

    case TokenKind::For():
      return ParseForStatement();

    case TokenKind::Continue():
      return ParseKeywordStatement(ParseNodeKind::ContinueStatement(),
                                   KeywordStatementArgument::None);

    case TokenKind::Break():
      return ParseKeywordStatement(ParseNodeKind::BreakStatement(),
                                   KeywordStatementArgument::None);

    case TokenKind::Return():
      return ParseKeywordStatement(ParseNodeKind::ReturnStatement(),
                                   KeywordStatementArgument::Optional);

    default:
      // A statement with no introducer token can only be an expression
      // statement.
      return ParseExpressionStatement();
  }
}

}  // namespace Carbon

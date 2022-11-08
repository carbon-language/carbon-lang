// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser2.h"

#include <cstdlib>
#include <memory>

#include "common/check.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// May be omitted a couple different ways as part of operator parsing.
CARBON_DIAGNOSTIC(
    OperatorRequiresParentheses, Error,
    "Parentheses are required to disambiguate operator precedence.");

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

class Parser2::PrettyStackTraceParseState : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceParseState(const Parser2* parser)
      : parser_(parser) {}
  ~PrettyStackTraceParseState() override = default;

  auto print(llvm::raw_ostream& output) const -> void override {
    output << "Parser stack:\n";
    for (int i = 0; i < static_cast<int>(parser_->state_stack_.size()); ++i) {
      const auto& entry = parser_->state_stack_[i];
      output << "\t" << i << ".\t" << entry.state;
      Print(output, entry.token);
    }
    output << "\tcursor\tposition_";
    Print(output, *parser_->position_);
  }

 private:
  auto Print(llvm::raw_ostream& output, TokenizedBuffer::Token token) const
      -> void {
    auto line = parser_->tokens_.GetLine(token);
    output << " @ " << parser_->tokens_.GetLineNumber(line) << ":"
           << parser_->tokens_.GetColumnNumber(token) << ":"
           << " token " << token << " : "
           << parser_->tokens_.GetKind(token).Name() << "\n";
  }

  const Parser2* parser_;
};

Parser2::Parser2(ParseTree& tree_arg, TokenizedBuffer& tokens_arg,
                 TokenDiagnosticEmitter& emitter)
    : tree_(tree_arg),
      tokens_(tokens_arg),
      emitter_(emitter),
      position_(tokens_.tokens().begin()),
      end_(tokens_.tokens().end()) {
  CARBON_CHECK(position_ != end_) << "Empty TokenizedBuffer";
  --end_;
  CARBON_CHECK(tokens_.GetKind(*end_) == TokenKind::EndOfFile())
      << "TokenizedBuffer should end with EndOfFile, ended with "
      << tokens_.GetKind(*end_).Name();
}

auto Parser2::AddLeafNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                          bool has_error) -> void {
  tree_.node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, /*subtree_size=*/1));
  if (has_error) {
    tree_.has_errors_ = true;
  }
}

auto Parser2::AddNode(ParseNodeKind kind, TokenizedBuffer::Token token,
                      int subtree_start, bool has_error) -> void {
  int subtree_size = tree_.size() - subtree_start + 1;
  tree_.node_impls_.push_back(
      ParseTree::NodeImpl(kind, has_error, token, subtree_size));
  if (has_error) {
    tree_.has_errors_ = true;
  }
}

auto Parser2::ConsumeAndAddCloseParen(TokenizedBuffer::Token open_paren,
                                      ParseNodeKind close_kind) -> bool {
  if (ConsumeAndAddLeafNodeIf(TokenKind::CloseParen(), close_kind)) {
    return true;
  }

  // TODO: Include the location of the matching open_paren in the diagnostic.
  CARBON_DIAGNOSTIC(ExpectedCloseParen, Error, "Unexpected tokens before `)`.");
  emitter_.Emit(*position_, ExpectedCloseParen);

  SkipTo(tokens_.GetMatchedClosingToken(open_paren));
  AddLeafNode(close_kind, *position_);
  ++position_;
  return false;
}

auto Parser2::ConsumeAndAddLeafNodeIf(TokenKind token_kind,
                                      ParseNodeKind node_kind) -> bool {
  auto token = ConsumeIf(token_kind);
  if (!token) {
    return false;
  }

  AddLeafNode(node_kind, *token);
  return true;
}

auto Parser2::ConsumeIf(TokenKind kind)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (!PositionIs(kind)) {
    return llvm::None;
  }
  auto token = *position_;
  ++position_;
  return token;
}

auto Parser2::FindNextOf(std::initializer_list<TokenKind> desired_kinds)
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

auto Parser2::SkipMatchingGroup() -> bool {
  if (!PositionKind().IsOpeningSymbol()) {
    return false;
  }

  SkipTo(tokens_.GetMatchedClosingToken(*position_));
  ++position_;
  return true;
}

auto Parser2::SkipPastLikelyEnd(TokenizedBuffer::Token skip_root)
    -> llvm::Optional<TokenizedBuffer::Token> {
  if (position_ == end_) {
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
    if (PositionIs(TokenKind::CloseCurlyBrace())) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return llvm::None;
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(TokenKind::Semi())) {
      return semi;
    }

    // Skip over any matching group of tokens_.
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position_;
  } while (position_ != end_ &&
           is_same_line_or_indent_greater_than_root(*position_));

  return llvm::None;
}

auto Parser2::SkipTo(TokenizedBuffer::Token t) -> void {
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards from " << position_
                                << " to " << TokenizedBuffer::TokenIterator(t);
  position_ = TokenizedBuffer::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

auto Parser2::HandleCodeBlock() -> void {
  PushState(ParserState::CodeBlockFinish());
  if (ConsumeAndAddLeafNodeIf(TokenKind::OpenCurlyBrace(),
                              ParseNodeKind::CodeBlockStart())) {
    PushState(ParserState::StatementScope());
  } else {
    AddLeafNode(ParseNodeKind::CodeBlockStart(), *position_,
                /*has_error=*/true);

    // Recover by parsing a single statement.
    CARBON_DIAGNOSTIC(ExpectedCodeBlock, Error, "Expected braced code block.");
    emitter_.Emit(*position_, ExpectedCodeBlock);

    HandleStatement(PositionKind());
  }
}

// Determines whether the given token is considered to be the start of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedStartOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::OpenParen(), TokenKind::Identifier(),
                       TokenKind::IntegerLiteral(), TokenKind::RealLiteral(),
                       TokenKind::StringLiteral()});
}

// Determines whether the given token is considered to be the end of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedEndOfOperand(TokenKind kind) -> bool {
  return kind.IsOneOf({TokenKind::CloseParen(), TokenKind::CloseCurlyBrace(),
                       TokenKind::CloseSquareBracket(), TokenKind::Identifier(),
                       TokenKind::IntegerLiteral(), TokenKind::RealLiteral(),
                       TokenKind::StringLiteral()});
}

// Determines whether the given token could possibly be the start of an
// operand. This is conservatively correct, and will never incorrectly return
// `false`, but can incorrectly return `true`.
static auto IsPossibleStartOfOperand(TokenKind kind) -> bool {
  return !kind.IsOneOf({TokenKind::CloseParen(), TokenKind::CloseCurlyBrace(),
                        TokenKind::CloseSquareBracket(), TokenKind::Comma(),
                        TokenKind::Semi(), TokenKind::Colon()});
}

auto Parser2::IsLexicallyValidInfixOperator() -> bool {
  CARBON_CHECK(position_ != end_) << "Expected an operator token.";

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

auto Parser2::IsTrailingOperatorInfix() -> bool {
  if (position_ == end_) {
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

auto Parser2::DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
  if (fixity == OperatorFixity::Infix) {
    // Infix operators must satisfy the infix operator rules.
    if (!IsLexicallyValidInfixOperator()) {
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
    if (PositionKind().IsSymbol() &&
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
    if (IsLexicallyValidInfixOperator()) {
      CARBON_DIAGNOSTIC(UnaryOperatorRequiresWhitespace, Error,
                        "Whitespace is required {0} this unary operator.",
                        RelativeLocation);
      emitter_.Emit(
          *position_, UnaryOperatorRequiresWhitespace,
          prefix ? RelativeLocation::Before : RelativeLocation::After);
    }
  }
}

auto Parser2::Parse() -> void {
  // Traces state_stack_. This runs even in opt because it's low overhead.
  PrettyStackTraceParseState pretty_stack(this);

  PushState(ParserState::Declaration());
  while (!state_stack_.empty()) {
    switch (state_stack_.back().state) {
#define CARBON_PARSER_STATE(Name) \
  case ParserState::Name():       \
    Handle##Name##State();        \
    break;
#include "toolchain/parser/parser_state.def"
    }
  }

  AddLeafNode(ParseNodeKind::FileEnd(), *position_);
}

auto Parser2::HandleCodeBlockFinishState() -> void {
  auto state = PopState();

  // If the block started with an open curly, this is a close curly.
  if (tokens_.GetKind(state.token) == TokenKind::OpenCurlyBrace()) {
    AddNode(ParseNodeKind::CodeBlock(), *position_, state.subtree_start,
            state.has_error);
    ++position_;
  } else {
    AddNode(ParseNodeKind::CodeBlock(), state.token, state.subtree_start,
            /*has_error=*/true);
  }
}

auto Parser2::HandleDeclarationState() -> void {
  // This maintains the current state unless we're at the end of the file.

  switch (PositionKind()) {
    case TokenKind::EndOfFile(): {
      PopAndDiscardState();
      break;
    }
    case TokenKind::Fn(): {
      PushState(ParserState::FunctionIntroducer());
      AddLeafNode(ParseNodeKind::FunctionIntroducer(), *position_);
      ++position_;
      break;
    }
    case TokenKind::Semi(): {
      AddLeafNode(ParseNodeKind::EmptyDeclaration(), *position_);
      ++position_;
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(UnrecognizedDeclaration, Error,
                        "Unrecognized declaration introducer.");
      emitter_.Emit(*position_, UnrecognizedDeclaration);
      tree_.has_errors_ = true;
      if (auto semi = SkipPastLikelyEnd(*position_)) {
        AddLeafNode(ParseNodeKind::EmptyDeclaration(), *semi,
                    /*has_error=*/true);
      }
      break;
    }
  }
}

/*
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
*/

auto Parser2::HandleExpressionInPostfixState() -> void {
  PopAndDiscardState();

  // Regardless of success or failure, we'll continue to the Resume state.
  PushState(ParserState::ExpressionInPostfixResume());

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  // TODO: Handle OpenParen and OpenCurlyBrace.
  switch (PositionKind()) {
    case TokenKind::Identifier():
      AddLeafNode(ParseNodeKind::NameReference(), *position_);
      break;

    case TokenKind::IntegerLiteral():
    case TokenKind::RealLiteral():
    case TokenKind::StringLiteral():
    case TokenKind::IntegerTypeLiteral():
    case TokenKind::UnsignedIntegerTypeLiteral():
    case TokenKind::FloatingPointTypeLiteral():
      AddLeafNode(ParseNodeKind::Literal(), *position_);
      break;

    default:
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      emitter_.Emit(*position_, ExpectedExpression);
      ReturnErrorOnState();
      return;
  }
  ++position_;
}

auto Parser2::HandleExpressionInPostfixResumeState() -> void {
  // This is a cyclic state that repeats, so the state is not popped.

  switch (PositionKind()) {
    case TokenKind::Period():
      // expression = ParseDesignatorExpression(
      //     start, ParseNodeKind::DesignatorExpression(), !expression);
      CARBON_FATAL() << "TODO";
      break;

    case TokenKind::OpenParen():
      // expression = ParseCallExpression(start, !expression);
      CARBON_FATAL() << "TODO";
      break;

    default:
      PopAndDiscardState();
      break;
  }
}

auto Parser2::HandleExpressionAsOperator(PrecedenceGroup ambient_precedence)
    -> void {
  // Check for a prefix operator.
  if (auto operator_precedence = PrecedenceGroup::ForLeading(PositionKind())) {
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

    PushStateForExpression(ParserState::ExpressionResumeForPrefix(),
                           ambient_precedence, *operator_precedence);
    ++position_;
    PushState(ParserState::Expression());
  } else {
    PushStateForExpression(ParserState::ExpressionResume(), ambient_precedence,
                           PrecedenceGroup::ForPostfixExpression());
    PushState(ParserState::ExpressionInPostfix());
  }
}

auto Parser2::HandleExpressionResumeState() -> void {
  auto state = PopState();

  auto trailing_operator =
      PrecedenceGroup::ForTrailing(PositionKind(), IsTrailingOperatorInfix());
  if (!trailing_operator) {
    return;
  }
  auto [operator_precedence, is_binary] = *trailing_operator;

  // TODO: If this operator is ambiguous with either the ambient precedence
  // or the LHS precedence, and there's a variant with a different fixity
  // that would work, use that one instead for error recovery.
  if (PrecedenceGroup::GetPriority(state.ambient_precedence,
                                   operator_precedence) !=
      OperatorPriority::RightFirst) {
    // The precedence rules don't permit this operator in this context. Try
    // again in the enclosing expression context.
    return;
  }

  if (PrecedenceGroup::GetPriority(state.lhs_precedence, operator_precedence) !=
      OperatorPriority::LeftFirst) {
    // Either the LHS operator and this operator are ambiguous, or the
    // LHS operator is a unary operator that can't be nested within
    // this operator. Either way, parentheses are required.
    emitter_.Emit(*position_, OperatorRequiresParentheses);
    state.has_error = true;
  } else {
    DiagnoseOperatorFixity(is_binary ? OperatorFixity::Infix
                                     : OperatorFixity::Postfix);
  }

  state.token = *position_;
  ++position_;

  state.lhs_precedence = operator_precedence;

  if (is_binary) {
    state.state = ParserState::ExpressionResumeForBinary();
    PushState(state);
    PushStateForExpression(ParserState::Expression(), operator_precedence,
                           state.lhs_precedence);
  } else {
    PushState(state);
    AddNode(ParseNodeKind::PostfixOperator(), state.token, state.subtree_start,
            state.has_error);
  }
}

auto Parser2::HandleExpressionResumeForBinaryState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::InfixOperator(), state.token, state.subtree_start,
          state.has_error);
  state.state = ParserState::ExpressionResume();
  PushState(state);
}

auto Parser2::HandleExpressionResumeForPrefixState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::PrefixOperator(), state.token, state.subtree_start,
          state.has_error);
  state.state = ParserState::ExpressionResume();
  PushState(state);
}

auto Parser2::HandleExpressionState() -> void {
  // TODO: This is temporary, we should need this state. If not, maybe add an
  // overload that uses pop_back instead of pop_back_val.
  auto state = PopState();

  HandleExpressionAsOperator(state.ambient_precedence);
}

auto Parser2::HandleExpressionForTypeState() -> void {
  // TODO: This is temporary, we should need this state. If not, maybe add an
  // overload that uses pop_back instead of pop_back_val.
  auto state = PopState();
  (void)state;

  HandleExpressionAsOperator(PrecedenceGroup::ForType());
}

auto Parser2::HandleExpressionStatementFinishState() -> void {
  auto state = PopState();

  if (auto semi = ConsumeIf(TokenKind::Semi())) {
    AddNode(ParseNodeKind::ExpressionStatement(), *semi, state.subtree_start,
            state.has_error);
    return;
  }

  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfterExpression, Error,
                      "Expected `;` after expression.");
    emitter_.Emit(*position_, ExpectedSemiAfterExpression);
  }

  if (auto semi_token = SkipPastLikelyEnd(state.token)) {
    AddNode(ParseNodeKind::ExpressionStatement(), *semi_token,
            state.subtree_start,
            /*has_error=*/true);
    return;
  }

  // Found junk not even followed by a `;`, no node to add.
  ReturnErrorOnState();
}

auto Parser2::HandleFunctionError(StateStackEntry state,
                                  bool skip_past_likely_end) -> void {
  auto token = state.token;
  if (skip_past_likely_end) {
    if (auto semi = SkipPastLikelyEnd(token)) {
      token = *semi;
    }
  }
  AddNode(ParseNodeKind::FunctionDeclaration(), token, state.subtree_start,
          /*has_error=*/true);
}

auto Parser2::HandleFunctionIntroducerState() -> void {
  auto state = PopState();

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Identifier(),
                               ParseNodeKind::DeclaredName())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionName, Error,
                      "Expected function name after `fn` keyword.");
    emitter_.Emit(*position_, ExpectedFunctionName);
    // TODO: We could change the lexer to allow us to synthesize certain
    // kinds of tokens and try to "recover" here, but unclear that this is
    // really useful.
    HandleFunctionError(state, true);
    return;
  }

  if (!PositionIs(TokenKind::OpenParen())) {
    CARBON_DIAGNOSTIC(ExpectedFunctionParams, Error,
                      "Expected `(` after function name.");
    emitter_.Emit(*position_, ExpectedFunctionParams);
    HandleFunctionError(state, true);
    return;
  }

  // Parse the parameter list as its own subtree; once that pops, resume
  // function parsing.
  state.state = ParserState::FunctionAfterParameterList();
  PushState(state);
  // TODO: When swapping () start/end, this should AddLeafNode the open before
  // continuing.
  PushState(ParserState::FunctionParameterListFinish());
  // Advance past the open paren.
  ++position_;
  if (PositionKind() != TokenKind::CloseParen()) {
    PushState(ParserState::PatternForFunctionParameter());
  }
}

auto Parser2::HandleFunctionParameterListFinishState() -> void {
  auto state = PopState();

  CARBON_CHECK(PositionKind() == TokenKind::CloseParen())
      << PositionKind().Name();
  AddLeafNode(ParseNodeKind::ParameterListEnd(), *position_);
  AddNode(ParseNodeKind::ParameterList(), state.token, state.subtree_start,
          state.has_error);
  ++position_;
}

auto Parser2::HandleFunctionAfterParameterListState() -> void {
  auto state = PopState();

  // Regardless of whether there's a return type, we'll finish the signature.
  state.state = ParserState::FunctionSignatureFinish();
  PushState(state);

  // If there is a return type, parse the expression before adding the return
  // type nod.e
  if (PositionIs(TokenKind::MinusGreater())) {
    PushState(ParserState::FunctionReturnTypeFinish());
    ++position_;
    PushState(ParserState::ExpressionForType());
  }
}

auto Parser2::HandleFunctionReturnTypeFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::ReturnType(), state.token, state.subtree_start,
          state.has_error);
}

auto Parser2::HandleFunctionSignatureFinishState() -> void {
  auto state = PopState();

  switch (PositionKind()) {
    case TokenKind::Semi(): {
      AddNode(ParseNodeKind::FunctionDeclaration(), *position_,
              state.subtree_start, state.has_error);
      ++position_;
      break;
    }
    case TokenKind::OpenCurlyBrace(): {
      AddNode(ParseNodeKind::FunctionDefinitionStart(), *position_,
              state.subtree_start, state.has_error);
      ++position_;
      // Any error is recorded on the FunctionDefinitionStart.
      state.has_error = false;
      state.state = ParserState::FunctionDefinitionFinish();
      PushState(state);
      PushState(ParserState::StatementScope());
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(
          ExpectedFunctionBodyOrSemi, Error,
          "Expected function definition or `;` after function declaration.");
      emitter_.Emit(*position_, ExpectedFunctionBodyOrSemi);
      // Only need to skip if we've not already found a new line.
      bool skip_past_likely_end =
          tokens_.GetLine(*position_) == tokens_.GetLine(state.token);
      HandleFunctionError(state, skip_past_likely_end);
      break;
    }
  }
}

auto Parser2::HandleFunctionDefinitionFinishState() -> void {
  auto state = PopState();
  AddNode(ParseNodeKind::FunctionDefinition(), *position_, state.subtree_start,
          state.has_error);
  ++position_;
}

auto Parser2::HandlePatternStart(PatternKind pattern_kind) -> void {
  auto state = PopState();

  // Ensure the finish state always follows.
  switch (pattern_kind) {
    case PatternKind::Parameter: {
      state.state = ParserState::PatternForFunctionParameterFinish();
      break;
    }
    case PatternKind::Variable: {
      CARBON_FATAL() << "TODO";
      break;
    }
  }

  // Handle an invalid pattern introducer.
  if (!PositionIs(TokenKind::Identifier()) ||
      tokens_.GetKind(*(position_ + 1)) != TokenKind::Colon()) {
    switch (pattern_kind) {
      case PatternKind::Parameter: {
        CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                          "Expected parameter declaration.");
        emitter_.Emit(*position_, ExpectedParameterName);
        break;
      }
      case PatternKind::Variable: {
        CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                          "Expected pattern in `var` declaration.");
        emitter_.Emit(*position_, ExpectedVariableName);
        break;
      }
    }
    state.has_error = true;
    PushState(state);
    return;
  }

  // Switch the context token to the colon, so that it'll be used for the root
  // node.
  state.token = *(position_ + 1);
  PushState(state);
  PushState(ParserState::ExpressionForType());
  AddLeafNode(ParseNodeKind::DeclaredName(), *position_);
  position_ += 2;
}

auto Parser2::HandleParenConditionState() -> void {
  auto state = PopState();

  auto open_paren = ConsumeIf(TokenKind::OpenParen());
  if (open_paren) {
    state.token = *open_paren;
  } else {
    CARBON_DIAGNOSTIC(ExpectedParenAfter, Error, "Expected `(` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedParenAfter, tokens_.GetKind(state.token));
  }

  // TODO: This should be adding a ConditionStart here instead of ConditionEnd
  // later, so this does state modification instead of a simpler push.
  state.state = ParserState::ParenConditionFinish();
  PushState(state);
  PushState(ParserState::Expression());
}

auto Parser2::HandleParenConditionFinishState() -> void {
  auto state = PopState();

  if (tokens_.GetKind(state.token) != TokenKind::OpenParen()) {
    // Don't expect a matching closing paren if there wasn't an opening paren.
    // TODO: Should probably push nodes on this state in order to have the
    // condition wrapped, but it wasn't before, so not doing it for consistency.
    ReturnErrorOnState();
    return;
  }

  bool close_paren =
      ConsumeAndAddCloseParen(state.token, ParseNodeKind::ConditionEnd());

  return AddNode(ParseNodeKind::Condition(), state.token, state.subtree_start,
                 /*has_error=*/state.has_error || !close_paren);
}

auto Parser2::HandlePatternForFunctionParameterState() -> void {
  HandlePatternStart(PatternKind::Parameter);
}

auto Parser2::HandlePatternForFunctionParameterFinishState() -> void {
  auto state = PopState();

  // If an error was encountered, propagate it without adding a node.
  bool has_error = state.has_error;
  if (has_error) {
    ReturnErrorOnState();
  } else {
    // TODO: may need to mark has_error if !type.
    AddNode(ParseNodeKind::PatternBinding(), state.token, state.subtree_start,
            state.has_error);
  }

  // Handle tokens following a parameter.
  switch (PositionKind()) {
    case TokenKind::CloseParen(): {
      // Done with the parameter list.
      return;
    }
    case TokenKind::Comma(): {
      // Comma handling is after the switch.
      break;
    }
    default: {
      // Don't error twice for the same issue.
      if (!has_error) {
        CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                          "Expected `,` or `)`.");
        emitter_.Emit(*position_, UnexpectedTokenAfterListElement);
        ReturnErrorOnState();
      }

      // Recover from the invalid token.
      auto end_of_element =
          FindNextOf({TokenKind::Comma(), TokenKind::CloseParen()});
      // The lexer guarantees that parentheses are balanced.
      CARBON_CHECK(end_of_element) << "missing matching `)` for `(`";
      SkipTo(*end_of_element);
      if (PositionKind() == TokenKind::CloseParen()) {
        // Done with the parameter list.
        return;
      }
      // Comma handling is after the switch.
      break;
    }
  }

  // We are guaranteed to now be at a comma.
  AddLeafNode(ParseNodeKind::ParameterListComma(), *position_);
  ++position_;
  if (PositionKind() != TokenKind::CloseParen()) {
    PushState(ParserState::PatternForFunctionParameter());
  }
}

auto Parser2::HandleStatementBreakFinishState() -> void {
  HandleStatementKeywordFinish(TokenKind::Break(),
                               ParseNodeKind::BreakStatement());
}

auto Parser2::HandleStatementContinueFinishState() -> void {
  HandleStatementKeywordFinish(TokenKind::Continue(),
                               ParseNodeKind::ContinueStatement());
}

auto Parser2::HandleStatementIf() -> void {
  PushState(ParserState::StatementIfConditionFinish());
  PushState(ParserState::ParenCondition());
  ++position_;
}

auto Parser2::HandleStatementIfConditionFinishState() -> void {
  auto state = PopState();

  state.state = ParserState::StatementIfThenBlockFinish();
  PushState(state);
  HandleCodeBlock();
}

auto Parser2::HandleStatementIfThenBlockFinishState() -> void {
  auto state = PopState();

  if (ConsumeAndAddLeafNodeIf(TokenKind::Else(),
                              ParseNodeKind::IfStatementElse())) {
    state.state = ParserState::StatementIfElseBlockFinish();
    PushState(state);
    // `else if` is permitted as a special case.
    if (PositionIs(TokenKind::If())) {
      HandleStatementIf();
    } else {
      HandleCodeBlock();
    }
  } else {
    AddNode(ParseNodeKind::IfStatement(), state.token, state.subtree_start,
            state.has_error);
  }
}

auto Parser2::HandleStatementIfElseBlockFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::IfStatement(), state.token, state.subtree_start,
          state.has_error);
}

auto Parser2::HandleStatementKeywordFinish(TokenKind token_kind,
                                           ParseNodeKind node_kind) -> void {
  auto state = PopState();

  if (!ConsumeAndAddLeafNodeIf(TokenKind::Semi(),
                               ParseNodeKind::StatementEnd())) {
    CARBON_DIAGNOSTIC(ExpectedSemiAfter, Error, "Expected `;` after `{0}`.",
                      TokenKind);
    emitter_.Emit(*position_, ExpectedSemiAfter, token_kind);
    if (auto semi_token = SkipPastLikelyEnd(state.token)) {
      AddLeafNode(ParseNodeKind::StatementEnd(), *semi_token,
                  /*has_error=*/true);
    }
  }
  AddNode(node_kind, state.token, state.subtree_start, state.has_error);
}

auto Parser2::HandleStatementReturnFinishState() -> void {
  HandleStatementKeywordFinish(TokenKind::Return(),
                               ParseNodeKind::ReturnStatement());
}

auto Parser2::HandleStatementWhile() -> void {
  PushState(ParserState::StatementWhileConditionFinish());
  PushState(ParserState::ParenCondition());
  ++position_;
}

auto Parser2::HandleStatementWhileConditionFinishState() -> void {
  auto state = PopState();

  state.state = ParserState::StatementWhileBlockFinish();
  PushState(state);
  HandleCodeBlock();
}

auto Parser2::HandleStatementWhileBlockFinishState() -> void {
  auto state = PopState();

  AddNode(ParseNodeKind::WhileStatement(), state.token, state.subtree_start,
          state.has_error);
}

auto Parser2::HandleStatement(TokenKind token_kind) -> void {
  switch (token_kind) {
    case TokenKind::If(): {
      HandleStatementIf();
      break;
    }
    case TokenKind::Break(): {
      PushState(ParserState::StatementBreakFinish());
      ++position_;
      break;
    }
    case TokenKind::Continue(): {
      PushState(ParserState::StatementContinueFinish());
      ++position_;
      break;
    }
    case TokenKind::Return(): {
      auto return_token = *position_;
      if (tokens_.GetKind(*(position_ + 1)) == TokenKind::Semi()) {
        int subtree_start = tree_.size();
        AddLeafNode(ParseNodeKind::StatementEnd(), *(position_ + 1));
        AddNode(ParseNodeKind::ReturnStatement(), return_token, subtree_start,
                /*has_error=*/false);
        position_ += 2;
      } else {
        PushState(ParserState::StatementReturnFinish());
        ++position_;
        PushState(ParserState::Expression());
      }
      break;
    }
    case TokenKind::While(): {
      HandleStatementWhile();
      break;
    }
    default: {
      PushState(ParserState::ExpressionStatementFinish());
      PushState(ParserState::Expression());
      break;
    }
  }
}

auto Parser2::HandleStatementScopeState() -> void {
  // This maintains the current state until we're at the end of the scope.

  auto token_kind = PositionKind();
  if (token_kind == TokenKind::CloseCurlyBrace()) {
    auto state = PopState();
    if (state.has_error) {
      ReturnErrorOnState();
    }
  } else {
    HandleStatement(token_kind);
  }
}

}  // namespace Carbon

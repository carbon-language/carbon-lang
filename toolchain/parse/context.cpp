// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

#include <optional>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/state.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// A relative location for characters in errors.
enum class RelativeLoc : int8_t {
  Around,
  After,
  Before,
};

}  // namespace Carbon::Parse

// Adapts RelativeLoc for use with formatv.
template <>
struct llvm::format_provider<Carbon::Parse::RelativeLoc> {
  using RelativeLoc = Carbon::Parse::RelativeLoc;
  static void format(const RelativeLoc& loc, raw_ostream& out,
                     StringRef /*style*/) {
    switch (loc) {
      case RelativeLoc::Around:
        out << "around";
        break;
      case RelativeLoc::After:
        out << "after";
        break;
      case RelativeLoc::Before:
        out << "before";
        break;
    }
  }
};

namespace Carbon::Parse {

Context::Context(Tree& tree, Lex::TokenizedBuffer& tokens,
                 Lex::TokenDiagnosticEmitter& emitter,
                 llvm::raw_ostream* vlog_stream)
    : tree_(&tree),
      tokens_(&tokens),
      emitter_(&emitter),
      vlog_stream_(vlog_stream),
      position_(tokens_->tokens().begin()),
      end_(tokens_->tokens().end()) {
  CARBON_CHECK(position_ != end_) << "Empty TokenizedBuffer";
  --end_;
  CARBON_CHECK(tokens_->GetKind(*end_) == Lex::TokenKind::FileEnd)
      << "TokenizedBuffer should end with FileEnd, ended with "
      << tokens_->GetKind(*end_);
}

auto Context::AddLeafNode(NodeKind kind, Lex::TokenIndex token, bool has_error)
    -> void {
  kind.CheckMatchesTokenKind(tokens_->GetKind(token), has_error);
  tree_->node_impls_.push_back(
      Tree::NodeImpl(kind, has_error, token, /*subtree_size=*/1));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto Context::AddNode(NodeKind kind, Lex::TokenIndex token, int subtree_start,
                      bool has_error) -> void {
  kind.CheckMatchesTokenKind(tokens_->GetKind(token), has_error);
  int subtree_size = tree_->size() - subtree_start + 1;
  tree_->node_impls_.push_back(
      Tree::NodeImpl(kind, has_error, token, subtree_size));
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto Context::ReplacePlaceholderNode(int32_t position, NodeKind kind,
                                     Lex::TokenIndex token, bool has_error)
    -> void {
  CARBON_CHECK(position >= 0 && position < tree_->size())
      << "position: " << position << " size: " << tree_->size();
  auto* node_impl = &tree_->node_impls_[position];
  CARBON_CHECK(node_impl->subtree_size == 1);
  CARBON_CHECK(node_impl->kind == NodeKind::Placeholder);
  node_impl->kind = kind;
  node_impl->has_error = has_error;
  node_impl->token = token;
  if (has_error) {
    tree_->has_errors_ = true;
  }
}

auto Context::ConsumeAndAddOpenParen(Lex::TokenIndex default_token,
                                     NodeKind start_kind)
    -> std::optional<Lex::TokenIndex> {
  if (auto open_paren = ConsumeIf(Lex::TokenKind::OpenParen)) {
    AddLeafNode(start_kind, *open_paren, /*has_error=*/false);
    return open_paren;
  } else {
    CARBON_DIAGNOSTIC(ExpectedParenAfter, Error, "Expected `(` after `{0}`.",
                      Lex::TokenKind);
    emitter_->Emit(*position_, ExpectedParenAfter,
                   tokens().GetKind(default_token));
    AddLeafNode(start_kind, default_token, /*has_error=*/true);
    return std::nullopt;
  }
}

auto Context::ConsumeAndAddCloseSymbol(Lex::TokenIndex expected_open,
                                       StateStackEntry state,
                                       NodeKind close_kind) -> void {
  Lex::TokenKind open_token_kind = tokens().GetKind(expected_open);

  if (!open_token_kind.is_opening_symbol()) {
    AddNode(close_kind, state.token, state.subtree_start, /*has_error=*/true);
  } else if (auto close_token = ConsumeIf(open_token_kind.closing_symbol())) {
    AddNode(close_kind, *close_token, state.subtree_start, state.has_error);
  } else {
    // TODO: Include the location of the matching opening delimiter in the
    // diagnostic.
    CARBON_DIAGNOSTIC(ExpectedCloseSymbol, Error,
                      "Unexpected tokens before `{0}`.", llvm::StringLiteral);
    emitter_->Emit(*position_, ExpectedCloseSymbol,
                   open_token_kind.closing_symbol().fixed_spelling());

    SkipTo(tokens().GetMatchedClosingToken(expected_open));
    AddNode(close_kind, Consume(), state.subtree_start, /*has_error=*/true);
  }
}

auto Context::ConsumeAndAddLeafNodeIf(Lex::TokenKind token_kind,
                                      NodeKind node_kind) -> bool {
  auto token = ConsumeIf(token_kind);
  if (!token) {
    return false;
  }

  AddLeafNode(node_kind, *token);
  return true;
}

auto Context::ConsumeChecked(Lex::TokenKind kind) -> Lex::TokenIndex {
  CARBON_CHECK(PositionIs(kind))
      << "Required " << kind << ", found " << PositionKind();
  return Consume();
}

auto Context::ConsumeIf(Lex::TokenKind kind) -> std::optional<Lex::TokenIndex> {
  if (!PositionIs(kind)) {
    return std::nullopt;
  }
  return Consume();
}

auto Context::FindNextOf(std::initializer_list<Lex::TokenKind> desired_kinds)
    -> std::optional<Lex::TokenIndex> {
  auto new_position = position_;
  while (true) {
    Lex::TokenIndex token = *new_position;
    Lex::TokenKind kind = tokens().GetKind(token);
    if (kind.IsOneOf(desired_kinds)) {
      return token;
    }

    // Step to the next token at the current bracketing level.
    if (kind.is_closing_symbol() || kind == Lex::TokenKind::FileEnd) {
      // There are no more tokens at this level.
      return std::nullopt;
    } else if (kind.is_opening_symbol()) {
      new_position = Lex::TokenIterator(tokens().GetMatchedClosingToken(token));
      // Advance past the closing token.
      ++new_position;
    } else {
      ++new_position;
    }
  }
}

auto Context::SkipMatchingGroup() -> bool {
  if (!PositionKind().is_opening_symbol()) {
    return false;
  }

  SkipTo(tokens().GetMatchedClosingToken(*position_));
  ++position_;
  return true;
}

auto Context::SkipPastLikelyEnd(Lex::TokenIndex skip_root) -> Lex::TokenIndex {
  if (position_ == end_) {
    return *(position_ - 1);
  }

  Lex::LineIndex root_line = tokens().GetLine(skip_root);
  int root_line_indent = tokens().GetIndentColumnNumber(root_line);

  // We will keep scanning through tokens on the same line as the root or
  // lines with greater indentation than root's line.
  auto is_same_line_or_indent_greater_than_root = [&](Lex::TokenIndex t) {
    Lex::LineIndex l = tokens().GetLine(t);
    if (l == root_line) {
      return true;
    }

    return tokens().GetIndentColumnNumber(l) > root_line_indent;
  };

  do {
    if (PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
      // Immediately bail out if we hit an unmatched close curly, this will
      // pop us up a level of the syntax grouping.
      return *(position_ - 1);
    }

    // We assume that a semicolon is always intended to be the end of the
    // current construct.
    if (auto semi = ConsumeIf(Lex::TokenKind::Semi)) {
      return *semi;
    }

    // Skip over any matching group of tokens().
    if (SkipMatchingGroup()) {
      continue;
    }

    // Otherwise just step forward one token.
    ++position_;
  } while (position_ != end_ &&
           is_same_line_or_indent_greater_than_root(*position_));

  return *(position_ - 1);
}

auto Context::SkipTo(Lex::TokenIndex t) -> void {
  CARBON_CHECK(t >= *position_) << "Tried to skip backwards from " << position_
                                << " to " << Lex::TokenIterator(t);
  position_ = Lex::TokenIterator(t);
  CARBON_CHECK(position_ != end_) << "Skipped past EOF.";
}

// Determines whether the given token is considered to be the start of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedStartOfOperand(Lex::TokenKind kind) -> bool {
  return kind.IsOneOf({Lex::TokenKind::OpenParen, Lex::TokenKind::Identifier,
                       Lex::TokenKind::IntLiteral, Lex::TokenKind::RealLiteral,
                       Lex::TokenKind::StringLiteral});
}

// Determines whether the given token is considered to be the end of an
// operand according to the rules for infix operator parsing.
static auto IsAssumedEndOfOperand(Lex::TokenKind kind) -> bool {
  return kind.IsOneOf(
      {Lex::TokenKind::CloseParen, Lex::TokenKind::CloseCurlyBrace,
       Lex::TokenKind::CloseSquareBracket, Lex::TokenKind::Identifier,
       Lex::TokenKind::IntLiteral, Lex::TokenKind::RealLiteral,
       Lex::TokenKind::StringLiteral});
}

// Determines whether the given token could possibly be the start of an
// operand. This is conservatively correct, and will never incorrectly return
// `false`, but can incorrectly return `true`.
static auto IsPossibleStartOfOperand(Lex::TokenKind kind) -> bool {
  return !kind.IsOneOf(
      {Lex::TokenKind::CloseParen, Lex::TokenKind::CloseCurlyBrace,
       Lex::TokenKind::CloseSquareBracket, Lex::TokenKind::Comma,
       Lex::TokenKind::Semi, Lex::TokenKind::Colon});
}

auto Context::IsLexicallyValidInfixOperator() -> bool {
  CARBON_CHECK(position_ != end_) << "Expected an operator token.";

  bool leading_space = tokens().HasLeadingWhitespace(*position_);
  bool trailing_space = tokens().HasTrailingWhitespace(*position_);

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
  if (position_ == tokens().tokens().begin() ||
      !IsAssumedEndOfOperand(tokens().GetKind(*(position_ - 1))) ||
      !IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return false;
  }

  return true;
}

auto Context::IsTrailingOperatorInfix() -> bool {
  if (position_ == end_) {
    return false;
  }

  // An operator that follows the infix operator rules is parsed as
  // infix, unless the next token means that it can't possibly be.
  if (IsLexicallyValidInfixOperator() &&
      IsPossibleStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return true;
  }

  // A trailing operator with leading whitespace that's not valid as infix is
  // not valid at all. If the next token looks like the start of an operand,
  // then parse as infix, otherwise as postfix. Either way we'll produce a
  // diagnostic later on.
  if (tokens().HasLeadingWhitespace(*position_) &&
      IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
    return true;
  }

  return false;
}

auto Context::DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
  if (!PositionKind().is_symbol()) {
    // Whitespace-based fixity rules only apply to symbolic operators.
    return;
  }

  if (fixity == OperatorFixity::Infix) {
    // Infix operators must satisfy the infix operator rules.
    if (!IsLexicallyValidInfixOperator()) {
      CARBON_DIAGNOSTIC(BinaryOperatorRequiresWhitespace, Error,
                        "Whitespace missing {0} binary operator.", RelativeLoc);
      emitter_->Emit(*position_, BinaryOperatorRequiresWhitespace,
                     tokens().HasLeadingWhitespace(*position_)
                         ? RelativeLoc::After
                         : (tokens().HasTrailingWhitespace(*position_)
                                ? RelativeLoc::Before
                                : RelativeLoc::Around));
    }
  } else {
    bool prefix = fixity == OperatorFixity::Prefix;

    // Whitespace is not permitted between a symbolic pre/postfix operator and
    // its operand.
    if ((prefix ? tokens().HasTrailingWhitespace(*position_)
                : tokens().HasLeadingWhitespace(*position_))) {
      CARBON_DIAGNOSTIC(UnaryOperatorHasWhitespace, Error,
                        "Whitespace is not allowed {0} this unary operator.",
                        RelativeLoc);
      emitter_->Emit(*position_, UnaryOperatorHasWhitespace,
                     prefix ? RelativeLoc::After : RelativeLoc::Before);
    } else if (IsLexicallyValidInfixOperator()) {
      // Pre/postfix operators must not satisfy the infix operator rules.
      CARBON_DIAGNOSTIC(UnaryOperatorRequiresWhitespace, Error,
                        "Whitespace is required {0} this unary operator.",
                        RelativeLoc);
      emitter_->Emit(*position_, UnaryOperatorRequiresWhitespace,
                     prefix ? RelativeLoc::Before : RelativeLoc::After);
    }
  }
}

auto Context::ConsumeListToken(NodeKind comma_kind, Lex::TokenKind close_kind,
                               bool already_has_error) -> ListTokenKind {
  if (!PositionIs(Lex::TokenKind::Comma) && !PositionIs(close_kind)) {
    // Don't error a second time on the same element.
    if (!already_has_error) {
      CARBON_DIAGNOSTIC(UnexpectedTokenAfterListElement, Error,
                        "Expected `,` or `{0}`.", Lex::TokenKind);
      emitter_->Emit(*position_, UnexpectedTokenAfterListElement, close_kind);
      ReturnErrorOnState();
    }

    // Recover from the invalid token.
    auto end_of_element = FindNextOf({Lex::TokenKind::Comma, close_kind});
    // The lexer guarantees that parentheses are balanced.
    CARBON_CHECK(end_of_element)
        << "missing matching `" << close_kind.opening_symbol() << "` for `"
        << close_kind << "`";

    SkipTo(*end_of_element);
  }

  if (PositionIs(close_kind)) {
    return ListTokenKind::Close;
  } else {
    AddLeafNode(comma_kind, Consume());
    return PositionIs(close_kind) ? ListTokenKind::CommaClose
                                  : ListTokenKind::Comma;
  }
}

auto Context::RecoverFromDeclError(StateStackEntry state, NodeKind node_kind,
                                   bool skip_past_likely_end) -> void {
  auto token = state.token;
  if (skip_past_likely_end) {
    token = SkipPastLikelyEnd(token);
  }
  AddNode(node_kind, token, state.subtree_start,
          /*has_error=*/true);
}

auto Context::DiagnoseExpectedDeclSemi(Lex::TokenKind expected_kind) -> void {
  CARBON_DIAGNOSTIC(ExpectedDeclSemi, Error,
                    "`{0}` declarations must end with a `;`.", Lex::TokenKind);
  emitter().Emit(*position(), ExpectedDeclSemi, expected_kind);
}

auto Context::DiagnoseExpectedDeclSemiOrDefinition(Lex::TokenKind expected_kind)
    -> void {
  CARBON_DIAGNOSTIC(ExpectedDeclSemiOrDefinition, Error,
                    "`{0}` declarations must either end with a `;` or "
                    "have a `{{ ... }` block for a definition.",
                    Lex::TokenKind);
  emitter().Emit(*position(), ExpectedDeclSemiOrDefinition, expected_kind);
}

// Returns whether we are currently parsing in a scope in which function
// definitions are deferred, such as a class or interface.
static auto ParsingInDeferredDefinitionScope(Context& context) -> bool {
  auto& stack = context.state_stack();
  if (stack.size() < 2 || stack.back().state != State::DeclScopeLoop) {
    return false;
  }
  auto state = stack[stack.size() - 2].state;
  return state == State::DeclDefinitionFinishAsClass ||
         state == State::DeclDefinitionFinishAsImpl ||
         state == State::DeclDefinitionFinishAsInterface ||
         state == State::DeclDefinitionFinishAsNamedConstraint;
}

auto Context::AddFunctionDefinitionStart(Lex::TokenIndex token,
                                         int subtree_start, bool has_error)
    -> void {
  if (ParsingInDeferredDefinitionScope(*this)) {
    enclosing_deferred_definition_stack_.push_back(
        tree_->deferred_definitions_.Add(
            {.start_id = FunctionDefinitionStartId(
                 NodeId(tree_->node_impls_.size()))}));
  }

  AddNode(NodeKind::FunctionDefinitionStart, token, subtree_start, has_error);
}

auto Context::AddFunctionDefinition(Lex::TokenIndex token, int subtree_start,
                                    bool has_error) -> void {
  if (ParsingInDeferredDefinitionScope(*this)) {
    auto definition_index = enclosing_deferred_definition_stack_.pop_back_val();
    auto& definition = tree_->deferred_definitions_.Get(definition_index);
    definition.definition_id =
        FunctionDefinitionId(NodeId(tree_->node_impls_.size()));
    definition.next_definition_index =
        DeferredDefinitionIndex(tree_->deferred_definitions().size());
  }

  AddNode(NodeKind::FunctionDefinition, token, subtree_start, has_error);
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  output << "Parser stack:\n";
  for (auto [i, entry] : llvm::enumerate(state_stack_)) {
    output << "\t" << i << ".\t" << entry.state;
    PrintTokenForStackDump(output, entry.token);
  }
  output << "\tcursor\tposition_";
  PrintTokenForStackDump(output, *position_);
}

auto Context::PrintTokenForStackDump(llvm::raw_ostream& output,
                                     Lex::TokenIndex token) const -> void {
  output << " @ " << tokens_->GetLineNumber(tokens_->GetLine(token)) << ":"
         << tokens_->GetColumnNumber(token) << ": token " << token << " : "
         << tokens_->GetKind(token) << "\n";
}

}  // namespace Carbon::Parse

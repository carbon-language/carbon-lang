// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

static auto DiagnoseStatementOperatorAsSubexpression(ParserContext& context)
    -> void {
  CARBON_DIAGNOSTIC(StatementOperatorAsSubexpression, Error,
                    "Operator `{0}` can only be used as a complete statement.",
                    TokenKind);
  context.emitter().Emit(*context.position(), StatementOperatorAsSubexpression,
                         context.PositionKind());
}

auto ParserHandleExpression(ParserContext& context) -> void {
  auto state = context.PopState();

  // Check for a prefix operator.
  if (auto operator_precedence =
          PrecedenceGroup::ForLeading(context.PositionKind())) {
    if (PrecedenceGroup::GetPriority(state.ambient_precedence,
                                     *operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this prefix operator in this
      // context. Diagnose this, but carry on and parse it anyway.
      if (PrecedenceGroup::GetPriority(PrecedenceGroup::ForTopLevelExpression(),
                                       *operator_precedence) ==
          OperatorPriority::RightFirst) {
        CARBON_DIAGNOSTIC(
            UnaryOperatorRequiresParentheses, Error,
            "Parentheses are required around this unary `{0}` operator.",
            TokenKind);
        context.emitter().Emit(*context.position(),
                               UnaryOperatorRequiresParentheses,
                               context.PositionKind());
      } else {
        // This operator wouldn't be allowed even if parenthesized.
        DiagnoseStatementOperatorAsSubexpression(context);
      }
    } else {
      // Check that this operator follows the proper whitespace rules.
      context.DiagnoseOperatorFixity(ParserContext::OperatorFixity::Prefix);
    }

    if (context.PositionIs(TokenKind::If)) {
      context.PushState(ParserState::IfExpressionFinish);
      context.PushState(ParserState::IfExpressionFinishCondition);
    } else {
      context.PushStateForExpressionLoop(ParserState::ExpressionLoopForPrefix,
                                         state.ambient_precedence,
                                         *operator_precedence);
    }

    ++context.position();
    context.PushStateForExpression(*operator_precedence);
  } else {
    context.PushStateForExpressionLoop(ParserState::ExpressionLoop,
                                       state.ambient_precedence,
                                       PrecedenceGroup::ForPostfixExpression());
    context.PushState(ParserState::ExpressionInPostfix);
  }
}

auto ParserHandleExpressionInPostfix(ParserContext& context) -> void {
  auto state = context.PopState();

  // Continue to the loop state.
  state.state = ParserState::ExpressionInPostfixLoop;

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  switch (context.PositionKind()) {
    case TokenKind::Identifier: {
      context.AddLeafNode(ParseNodeKind::NameExpression, context.Consume());
      context.PushState(state);
      break;
    }
    case TokenKind::False:
    case TokenKind::True:
    case TokenKind::IntegerLiteral:
    case TokenKind::RealLiteral:
    case TokenKind::StringLiteral:
    case TokenKind::Bool:
    case TokenKind::IntegerTypeLiteral:
    case TokenKind::UnsignedIntegerTypeLiteral:
    case TokenKind::FloatingPointTypeLiteral:
    case TokenKind::StringTypeLiteral:
    case TokenKind::Type: {
      context.AddLeafNode(ParseNodeKind::Literal, context.Consume());
      context.PushState(state);
      break;
    }
    case TokenKind::OpenCurlyBrace: {
      context.PushState(state);
      context.PushState(ParserState::BraceExpression);
      break;
    }
    case TokenKind::OpenParen: {
      context.PushState(state);
      context.PushState(ParserState::ParenExpression);
      break;
    }
    case TokenKind::OpenSquareBracket: {
      context.PushState(state);
      context.PushState(ParserState::ArrayExpression);
      break;
    }
    case TokenKind::SelfValueIdentifier: {
      context.AddLeafNode(ParseNodeKind::SelfValueName, context.Consume());
      context.PushState(state);
      break;
    }
    case TokenKind::SelfTypeIdentifier: {
      context.AddLeafNode(ParseNodeKind::SelfTypeNameExpression,
                          context.Consume());
      context.PushState(state);
      break;
    }
    default: {
      // Add a node to keep the parse tree balanced.
      context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                          /*has_error=*/true);
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      context.emitter().Emit(*context.position(), ExpectedExpression);
      context.ReturnErrorOnState();
      break;
    }
  }
}

auto ParserHandleExpressionInPostfixLoop(ParserContext& context) -> void {
  // This is a cyclic state that repeats, so this state is typically pushed back
  // on.
  auto state = context.PopState();
  state.token = *context.position();
  switch (context.PositionKind()) {
    case TokenKind::Period: {
      context.PushState(state);
      state.state = ParserState::PeriodAsExpression;
      context.PushState(state);
      break;
    }
    case TokenKind::MinusGreater: {
      context.PushState(state);
      state.state = ParserState::ArrowExpression;
      context.PushState(state);
      break;
    }
    case TokenKind::OpenParen: {
      context.PushState(state);
      state.state = ParserState::CallExpression;
      context.PushState(state);
      break;
    }
    case TokenKind::OpenSquareBracket: {
      context.PushState(state);
      state.state = ParserState::IndexExpression;
      context.PushState(state);
      break;
    }
    default: {
      if (state.has_error) {
        context.ReturnErrorOnState();
      }
      break;
    }
  }
}

auto ParserHandleExpressionLoop(ParserContext& context) -> void {
  auto state = context.PopState();

  auto operator_kind = context.PositionKind();
  auto trailing_operator = PrecedenceGroup::ForTrailing(
      operator_kind, context.IsTrailingOperatorInfix());
  if (!trailing_operator) {
    if (state.has_error) {
      context.ReturnErrorOnState();
    }
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
    if (state.has_error) {
      context.ReturnErrorOnState();
    }
    return;
  }

  if (PrecedenceGroup::GetPriority(state.lhs_precedence, operator_precedence) !=
      OperatorPriority::LeftFirst) {
    // Either the LHS operator and this operator are ambiguous, or the
    // LHS operator is a unary operator that can't be nested within
    // this operator. Either way, parentheses are required.
    if (PrecedenceGroup::GetPriority(PrecedenceGroup::ForTopLevelExpression(),
                                     operator_precedence) ==
        OperatorPriority::RightFirst) {
      CARBON_DIAGNOSTIC(
          OperatorRequiresParentheses, Error,
          "Parentheses are required to disambiguate operator precedence.");
      context.emitter().Emit(*context.position(), OperatorRequiresParentheses);
    } else {
      // This operator wouldn't be allowed even if parenthesized.
      DiagnoseStatementOperatorAsSubexpression(context);
    }
    state.has_error = true;
  } else {
    context.DiagnoseOperatorFixity(
        is_binary ? ParserContext::OperatorFixity::Infix
                  : ParserContext::OperatorFixity::Postfix);
  }

  state.token = context.Consume();
  state.lhs_precedence = operator_precedence;

  if (is_binary) {
    if (operator_kind == TokenKind::And || operator_kind == TokenKind::Or) {
      // For `and` and `or`, wrap the first operand in a virtual parse tree
      // node so that semantics can insert control flow here.
      context.AddNode(ParseNodeKind::ShortCircuitOperand, state.token,
                      state.subtree_start, state.has_error);
    }

    state.state = ParserState::ExpressionLoopForBinary;
    context.PushState(state);
    context.PushStateForExpression(operator_precedence);
  } else {
    context.AddNode(ParseNodeKind::PostfixOperator, state.token,
                    state.subtree_start, state.has_error);
    state.has_error = false;
    context.PushState(state);
  }
}

auto ParserHandleExpressionLoopForBinary(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::InfixOperator, state.token,
                  state.subtree_start, state.has_error);
  state.state = ParserState::ExpressionLoop;
  state.has_error = false;
  context.PushState(state);
}

auto ParserHandleExpressionLoopForPrefix(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::PrefixOperator, state.token,
                  state.subtree_start, state.has_error);
  state.state = ParserState::ExpressionLoop;
  state.has_error = false;
  context.PushState(state);
}

auto ParserHandleIfExpressionFinishCondition(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::IfExpressionIf, state.token,
                  state.subtree_start, state.has_error);

  if (context.PositionIs(TokenKind::Then)) {
    context.PushState(ParserState::IfExpressionFinishThen);
    context.ConsumeChecked(TokenKind::Then);
    context.PushStateForExpression(*PrecedenceGroup::ForLeading(TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedThenAfterIf, Error,
                      "Expected `then` after `if` condition.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedThenAfterIf);
    }
    // Add placeholders for `IfExpressionThen` and final `Expression`.
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto ParserHandleIfExpressionFinishThen(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::IfExpressionThen, state.token,
                  state.subtree_start, state.has_error);

  if (context.PositionIs(TokenKind::Else)) {
    context.PushState(ParserState::IfExpressionFinishElse);
    context.ConsumeChecked(TokenKind::Else);
    context.PushStateForExpression(*PrecedenceGroup::ForLeading(TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedElseAfterIf, Error,
                      "Expected `else` after `if ... then ...`.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedElseAfterIf);
    }
    // Add placeholder for the final `Expression`.
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto ParserHandleIfExpressionFinishElse(ParserContext& context) -> void {
  auto else_state = context.PopState();

  // Propagate the location of `else`.
  auto if_state = context.PopState();
  if_state.token = else_state.token;
  if_state.has_error |= else_state.has_error;
  context.PushState(if_state);
}

auto ParserHandleIfExpressionFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::IfExpressionElse, state.token,
                  state.subtree_start, state.has_error);
}

auto ParserHandleExpressionStatementFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  if (auto semi = context.ConsumeIf(TokenKind::Semi)) {
    context.AddNode(ParseNodeKind::ExpressionStatement, *semi,
                    state.subtree_start, state.has_error);
    return;
  }

  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ExpectedExpressionSemi, Error,
                      "Expected `;` after expression statement.");
    context.emitter().Emit(*context.position(), ExpectedExpressionSemi);
  }

  if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
    context.AddNode(ParseNodeKind::ExpressionStatement, *semi_token,
                    state.subtree_start,
                    /*has_error=*/true);
    return;
  }

  // Found junk not even followed by a `;`, no node to add.
  context.ReturnErrorOnState();
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

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
      context.emitter().Emit(*context.position(), OperatorRequiresParentheses);
    } else {
      // Check that this operator follows the proper whitespace rules.
      context.DiagnoseOperatorFixity(ParserContext::OperatorFixity::Prefix);
    }

    context.PushStateForExpressionLoop(ParserState::ExpressionLoopForPrefix,
                                       state.ambient_precedence,
                                       *operator_precedence);
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
      context.AddLeafNode(ParseNodeKind::NameReference, context.Consume());
      context.PushState(state);
      break;
    }
    case TokenKind::IntegerLiteral:
    case TokenKind::RealLiteral:
    case TokenKind::StringLiteral:
    case TokenKind::IntegerTypeLiteral:
    case TokenKind::UnsignedIntegerTypeLiteral:
    case TokenKind::FloatingPointTypeLiteral:
    case TokenKind::StringTypeLiteral: {
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
    case TokenKind::SelfValueIdentifier: {
      context.AddLeafNode(ParseNodeKind::SelfValueIdentifier,
                          context.Consume());
      context.PushState(state);
      break;
    }
    case TokenKind::SelfTypeIdentifier: {
      context.AddLeafNode(ParseNodeKind::SelfTypeIdentifier, context.Consume());
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
      state.state = ParserState::DesignatorAsExpression;
      context.PushState(state);
      break;
    }
    case TokenKind::OpenParen: {
      context.PushState(state);
      state.state = ParserState::CallExpression;
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

  auto trailing_operator = PrecedenceGroup::ForTrailing(
      context.PositionKind(), context.IsTrailingOperatorInfix());
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
    context.emitter().Emit(*context.position(), OperatorRequiresParentheses);
    state.has_error = true;
  } else {
    context.DiagnoseOperatorFixity(
        is_binary ? ParserContext::OperatorFixity::Infix
                  : ParserContext::OperatorFixity::Postfix);
  }

  state.token = context.Consume();
  state.lhs_precedence = operator_precedence;

  if (is_binary) {
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

auto ParserHandleExpressionStatementFinish(ParserContext& context) -> void {
  auto state = context.PopState();

  if (auto semi = context.ConsumeIf(TokenKind::Semi)) {
    context.AddNode(ParseNodeKind::ExpressionStatement, *semi,
                    state.subtree_start, state.has_error);
    return;
  }

  if (!state.has_error) {
    context.emitter().Emit(*context.position(), ExpectedSemiAfterExpression);
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

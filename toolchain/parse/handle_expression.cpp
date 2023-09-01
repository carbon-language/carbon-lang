// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto DiagnoseStatementOperatorAsSubexpression(Context& context) -> void {
  CARBON_DIAGNOSTIC(StatementOperatorAsSubexpression, Error,
                    "Operator `{0}` can only be used as a complete statement.",
                    Lex::TokenKind);
  context.emitter().Emit(*context.position(), StatementOperatorAsSubexpression,
                         context.PositionKind());
}

auto HandleExpression(Context& context) -> void {
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
            Lex::TokenKind);
        context.emitter().Emit(*context.position(),
                               UnaryOperatorRequiresParentheses,
                               context.PositionKind());
      } else {
        // This operator wouldn't be allowed even if parenthesized.
        DiagnoseStatementOperatorAsSubexpression(context);
      }
    } else {
      // Check that this operator follows the proper whitespace rules.
      context.DiagnoseOperatorFixity(Context::OperatorFixity::Prefix);
    }

    if (context.PositionIs(Lex::TokenKind::If)) {
      context.PushState(State::IfExpressionFinish);
      context.PushState(State::IfExpressionFinishCondition);
    } else {
      context.PushStateForExpressionLoop(State::ExpressionLoopForPrefix,
                                         state.ambient_precedence,
                                         *operator_precedence);
    }

    ++context.position();
    context.PushStateForExpression(*operator_precedence);
  } else {
    context.PushStateForExpressionLoop(State::ExpressionLoop,
                                       state.ambient_precedence,
                                       PrecedenceGroup::ForPostfixExpression());
    context.PushState(State::ExpressionInPostfix);
  }
}

auto HandleExpressionInPostfix(Context& context) -> void {
  auto state = context.PopState();

  // Continue to the loop state.
  state.state = State::ExpressionInPostfixLoop;

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  switch (context.PositionKind()) {
    case Lex::TokenKind::Identifier: {
      context.AddLeafNode(NodeKind::NameExpression, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::False:
    case Lex::TokenKind::True:
    case Lex::TokenKind::IntegerLiteral:
    case Lex::TokenKind::RealLiteral:
    case Lex::TokenKind::StringLiteral:
    case Lex::TokenKind::Bool:
    case Lex::TokenKind::IntegerTypeLiteral:
    case Lex::TokenKind::UnsignedIntegerTypeLiteral:
    case Lex::TokenKind::FloatingPointTypeLiteral:
    case Lex::TokenKind::StringTypeLiteral:
    case Lex::TokenKind::Type: {
      context.AddLeafNode(NodeKind::Literal, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::OpenCurlyBrace: {
      context.PushState(state);
      context.PushState(State::BraceExpression);
      break;
    }
    case Lex::TokenKind::OpenParen: {
      context.PushState(state);
      context.PushState(State::ParenExpression);
      break;
    }
    case Lex::TokenKind::OpenSquareBracket: {
      context.PushState(state);
      context.PushState(State::ArrayExpression);
      break;
    }
    case Lex::TokenKind::SelfValueIdentifier: {
      context.AddLeafNode(NodeKind::SelfValueName, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::SelfTypeIdentifier: {
      context.AddLeafNode(NodeKind::SelfTypeNameExpression, context.Consume());
      context.PushState(state);
      break;
    }
    default: {
      // Add a node to keep the parse tree balanced.
      context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                          /*has_error=*/true);
      CARBON_DIAGNOSTIC(ExpectedExpression, Error, "Expected expression.");
      context.emitter().Emit(*context.position(), ExpectedExpression);
      context.ReturnErrorOnState();
      break;
    }
  }
}

auto HandleExpressionInPostfixLoop(Context& context) -> void {
  // This is a cyclic state that repeats, so this state is typically pushed back
  // on.
  auto state = context.PopState();
  state.token = *context.position();
  switch (context.PositionKind()) {
    case Lex::TokenKind::Period: {
      context.PushState(state);
      state.state = State::PeriodAsExpression;
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::MinusGreater: {
      context.PushState(state);
      state.state = State::ArrowExpression;
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::OpenParen: {
      context.PushState(state);
      state.state = State::CallExpression;
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::OpenSquareBracket: {
      context.PushState(state);
      state.state = State::IndexExpression;
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

auto HandleExpressionLoop(Context& context) -> void {
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
    context.DiagnoseOperatorFixity(is_binary
                                       ? Context::OperatorFixity::Infix
                                       : Context::OperatorFixity::Postfix);
  }

  state.token = context.Consume();
  state.lhs_precedence = operator_precedence;

  if (is_binary) {
    if (operator_kind == Lex::TokenKind::And ||
        operator_kind == Lex::TokenKind::Or) {
      // For `and` and `or`, wrap the first operand in a virtual parse tree
      // node so that semantics can insert control flow here.
      context.AddNode(NodeKind::ShortCircuitOperand, state.token,
                      state.subtree_start, state.has_error);
    }

    state.state = State::ExpressionLoopForBinary;
    context.PushState(state);
    context.PushStateForExpression(operator_precedence);
  } else {
    context.AddNode(NodeKind::PostfixOperator, state.token, state.subtree_start,
                    state.has_error);
    state.has_error = false;
    context.PushState(state);
  }
}

auto HandleExpressionLoopForBinary(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::InfixOperator, state.token, state.subtree_start,
                  state.has_error);
  state.state = State::ExpressionLoop;
  state.has_error = false;
  context.PushState(state);
}

auto HandleExpressionLoopForPrefix(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::PrefixOperator, state.token, state.subtree_start,
                  state.has_error);
  state.state = State::ExpressionLoop;
  state.has_error = false;
  context.PushState(state);
}

auto HandleIfExpressionFinishCondition(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExpressionIf, state.token, state.subtree_start,
                  state.has_error);

  if (context.PositionIs(Lex::TokenKind::Then)) {
    context.PushState(State::IfExpressionFinishThen);
    context.ConsumeChecked(Lex::TokenKind::Then);
    context.PushStateForExpression(
        *PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedThenAfterIf, Error,
                      "Expected `then` after `if` condition.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedThenAfterIf);
    }
    // Add placeholders for `IfExpressionThen` and final `Expression`.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto HandleIfExpressionFinishThen(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExpressionThen, state.token, state.subtree_start,
                  state.has_error);

  if (context.PositionIs(Lex::TokenKind::Else)) {
    context.PushState(State::IfExpressionFinishElse);
    context.ConsumeChecked(Lex::TokenKind::Else);
    context.PushStateForExpression(
        *PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedElseAfterIf, Error,
                      "Expected `else` after `if ... then ...`.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedElseAfterIf);
    }
    // Add placeholder for the final `Expression`.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto HandleIfExpressionFinishElse(Context& context) -> void {
  auto else_state = context.PopState();

  // Propagate the location of `else`.
  auto if_state = context.PopState();
  if_state.token = else_state.token;
  if_state.has_error |= else_state.has_error;
  context.PushState(if_state);
}

auto HandleIfExpressionFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExpressionElse, state.token, state.subtree_start,
                  state.has_error);
}

auto HandleExpressionStatementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(NodeKind::ExpressionStatement, *semi, state.subtree_start,
                    state.has_error);
    return;
  }

  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ExpectedExpressionSemi, Error,
                      "Expected `;` after expression statement.");
    context.emitter().Emit(*context.position(), ExpectedExpressionSemi);
  }

  if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
    context.AddNode(NodeKind::ExpressionStatement, *semi_token,
                    state.subtree_start,
                    /*has_error=*/true);
    return;
  }

  // Found junk not even followed by a `;`, no node to add.
  context.ReturnErrorOnState();
}

}  // namespace Carbon::Parse

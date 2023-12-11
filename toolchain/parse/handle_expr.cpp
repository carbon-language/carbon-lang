// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

static auto DiagnoseStatementOperatorAsSubExpr(Context& context) -> void {
  CARBON_DIAGNOSTIC(StatementOperatorAsSubExpr, Error,
                    "Operator `{0}` can only be used as a complete statement.",
                    Lex::TokenKind);
  context.emitter().Emit(*context.position(), StatementOperatorAsSubExpr,
                         context.PositionKind());
}

auto HandleExpr(Context& context) -> void {
  auto state = context.PopState();

  // Check for a prefix operator.
  if (auto operator_precedence =
          PrecedenceGroup::ForLeading(context.PositionKind())) {
    if (PrecedenceGroup::GetPriority(state.ambient_precedence,
                                     *operator_precedence) !=
        OperatorPriority::RightFirst) {
      // The precedence rules don't permit this prefix operator in this
      // context. Diagnose this, but carry on and parse it anyway.
      if (PrecedenceGroup::GetPriority(PrecedenceGroup::ForTopLevelExpr(),
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
        DiagnoseStatementOperatorAsSubExpr(context);
      }
    } else {
      // Check that this operator follows the proper whitespace rules.
      context.DiagnoseOperatorFixity(Context::OperatorFixity::Prefix);
    }

    if (context.PositionIs(Lex::TokenKind::If)) {
      context.PushState(State::IfExprFinish);
      context.PushState(State::IfExprFinishCondition);
    } else {
      context.PushStateForExprLoop(State::ExprLoopForPrefix,
                                   state.ambient_precedence,
                                   *operator_precedence);
    }

    context.ConsumeAndDiscard();
    context.PushStateForExpr(*operator_precedence);
  } else {
    context.PushStateForExprLoop(State::ExprLoop, state.ambient_precedence,
                                 PrecedenceGroup::ForPostfixExpr());
    context.PushState(State::ExprInPostfix);
  }
}

auto HandleExprInPostfix(Context& context) -> void {
  auto state = context.PopState();

  // Continue to the loop state.
  state.state = State::ExprInPostfixLoop;

  // Parses a primary expression, which is either a terminal portion of an
  // expression tree, such as an identifier or literal, or a parenthesized
  // expression.
  switch (context.PositionKind()) {
    case Lex::TokenKind::Identifier: {
      context.AddLeafNode(NodeKind::IdentifierNameExpr, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::False: {
      context.AddLeafNode(NodeKind::BoolLiteralFalse, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::True: {
      context.AddLeafNode(NodeKind::BoolLiteralTrue, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::IntLiteral: {
      context.AddLeafNode(NodeKind::IntLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::RealLiteral: {
      context.AddLeafNode(NodeKind::RealLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::StringLiteral: {
      context.AddLeafNode(NodeKind::StringLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::Bool: {
      context.AddLeafNode(NodeKind::BoolTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::IntTypeLiteral: {
      context.AddLeafNode(NodeKind::IntTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::UnsignedIntTypeLiteral: {
      context.AddLeafNode(NodeKind::UnsignedIntTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::FloatTypeLiteral: {
      context.AddLeafNode(NodeKind::FloatTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::StringTypeLiteral: {
      context.AddLeafNode(NodeKind::StringTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::Type: {
      context.AddLeafNode(NodeKind::TypeTypeLiteral, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::OpenCurlyBrace: {
      context.PushState(state);
      context.PushState(State::BraceExpr);
      break;
    }
    case Lex::TokenKind::OpenParen: {
      context.PushState(state);
      context.PushState(State::ParenExpr);
      break;
    }
    case Lex::TokenKind::OpenSquareBracket: {
      context.PushState(state);
      context.PushState(State::ArrayExpr);
      break;
    }
    case Lex::TokenKind::Package: {
      context.AddLeafNode(NodeKind::PackageExpr, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::SelfValueIdentifier: {
      context.AddLeafNode(NodeKind::SelfValueNameExpr, context.Consume());
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::SelfTypeIdentifier: {
      context.AddLeafNode(NodeKind::SelfTypeNameExpr, context.Consume());
      context.PushState(state);
      break;
    }
    default: {
      // Add a node to keep the parse tree balanced.
      context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                          /*has_error=*/true);
      CARBON_DIAGNOSTIC(ExpectedExpr, Error, "Expected expression.");
      context.emitter().Emit(*context.position(), ExpectedExpr);
      context.ReturnErrorOnState();
      break;
    }
  }
}

auto HandleExprInPostfixLoop(Context& context) -> void {
  // This is a cyclic state that repeats, so this state is typically pushed back
  // on.
  auto state = context.PopState();
  state.token = *context.position();
  switch (context.PositionKind()) {
    case Lex::TokenKind::Period: {
      context.PushState(state);
      context.PushState(state, State::PeriodAsExpr);
      break;
    }
    case Lex::TokenKind::MinusGreater: {
      context.PushState(state);
      context.PushState(state, State::ArrowExpr);
      break;
    }
    case Lex::TokenKind::OpenParen: {
      context.PushState(state);
      context.PushState(state, State::CallExpr);
      break;
    }
    case Lex::TokenKind::OpenSquareBracket: {
      context.PushState(state);
      context.PushState(state, State::IndexExpr);
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

auto HandleExprLoop(Context& context) -> void {
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
    if (PrecedenceGroup::GetPriority(PrecedenceGroup::ForTopLevelExpr(),
                                     operator_precedence) ==
        OperatorPriority::RightFirst) {
      CARBON_DIAGNOSTIC(
          OperatorRequiresParentheses, Error,
          "Parentheses are required to disambiguate operator precedence.");
      context.emitter().Emit(*context.position(), OperatorRequiresParentheses);
    } else {
      // This operator wouldn't be allowed even if parenthesized.
      DiagnoseStatementOperatorAsSubExpr(context);
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
    switch (operator_kind) {
      // For `and` and `or`, wrap the first operand in a virtual parse tree
      // node so that checking can insert control flow here.
      case Lex::TokenKind::And:
        context.AddNode(NodeKind::ShortCircuitOperandAnd, state.token,
                        state.subtree_start, state.has_error);
        state.state = State::ExprLoopForShortCircuitOperatorAsAnd;
        break;
      case Lex::TokenKind::Or:
        context.AddNode(NodeKind::ShortCircuitOperandOr, state.token,
                        state.subtree_start, state.has_error);
        state.state = State::ExprLoopForShortCircuitOperatorAsOr;
        break;

      default:
        state.state = State::ExprLoopForBinary;
        break;
    }

    context.PushState(state);
    context.PushStateForExpr(operator_precedence);
  } else {
    context.AddNode(NodeKind::PostfixOperator, state.token, state.subtree_start,
                    state.has_error);
    state.has_error = false;
    context.PushState(state);
  }
}

// Adds the operator node and returns the the main expression loop.
static auto HandleExprLoopForOperator(Context& context, NodeKind node_kind)
    -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, state.token, state.subtree_start, state.has_error);
  state.has_error = false;
  context.PushState(state, State::ExprLoop);
}

auto HandleExprLoopForBinary(Context& context) -> void {
  HandleExprLoopForOperator(context, NodeKind::InfixOperator);
}

auto HandleExprLoopForPrefix(Context& context) -> void {
  HandleExprLoopForOperator(context, NodeKind::PrefixOperator);
}

auto HandleExprLoopForShortCircuitOperatorAsAnd(Context& context) -> void {
  HandleExprLoopForOperator(context, NodeKind::ShortCircuitOperatorAnd);
}

auto HandleExprLoopForShortCircuitOperatorAsOr(Context& context) -> void {
  HandleExprLoopForOperator(context, NodeKind::ShortCircuitOperatorOr);
}

auto HandleIfExprFinishCondition(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprIf, state.token, state.subtree_start,
                  state.has_error);

  if (context.PositionIs(Lex::TokenKind::Then)) {
    context.PushState(State::IfExprFinishThen);
    context.ConsumeChecked(Lex::TokenKind::Then);
    context.PushStateForExpr(*PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedThenAfterIf, Error,
                      "Expected `then` after `if` condition.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedThenAfterIf);
    }
    // Add placeholders for `IfExprThen` and final `Expr`.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto HandleIfExprFinishThen(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprThen, state.token, state.subtree_start,
                  state.has_error);

  if (context.PositionIs(Lex::TokenKind::Else)) {
    context.PushState(State::IfExprFinishElse);
    context.ConsumeChecked(Lex::TokenKind::Else);
    context.PushStateForExpr(*PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    // TODO: Include the location of the `if` token.
    CARBON_DIAGNOSTIC(ExpectedElseAfterIf, Error,
                      "Expected `else` after `if ... then ...`.");
    if (!state.has_error) {
      context.emitter().Emit(*context.position(), ExpectedElseAfterIf);
    }
    // Add placeholder for the final `Expr`.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  }
}

auto HandleIfExprFinishElse(Context& context) -> void {
  auto else_state = context.PopState();

  // Propagate the location of `else`.
  auto if_state = context.PopState();
  if_state.token = else_state.token;
  if_state.has_error |= else_state.has_error;
  context.PushState(if_state);
}

auto HandleIfExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprElse, state.token, state.subtree_start,
                  state.has_error);
}

auto HandleExprStatementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(NodeKind::ExprStatement, *semi, state.subtree_start,
                    state.has_error);
    return;
  }

  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ExpectedExprSemi, Error,
                      "Expected `;` after expression statement.");
    context.emitter().Emit(*context.position(), ExpectedExprSemi);
  }

  if (auto semi_token = context.SkipPastLikelyEnd(state.token)) {
    context.AddNode(NodeKind::ExprStatement, *semi_token, state.subtree_start,
                    /*has_error=*/true);
    return;
  }

  // Found junk not even followed by a `;`, no node to add.
  context.ReturnErrorOnState();
}

}  // namespace Carbon::Parse

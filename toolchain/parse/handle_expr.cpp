// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

static auto DiagnoseStatementOperatorAsSubExpr(Context& context) -> void {
  CARBON_DIAGNOSTIC(StatementOperatorAsSubExpr, Error,
                    "operator `{0}` can only be used as a complete statement",
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
            "parentheses are required around this unary `{0}` operator",
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
      context.PushStateForExprLoop(State::ExprLoopForPrefixOperator,
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
  switch (auto token_kind = context.PositionKind()) {
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
    case Lex::TokenKind::Auto: {
      context.AddLeafNode(NodeKind::AutoTypeLiteral, context.Consume());
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
    case Lex::TokenKind::Array: {
      context.PushState(state);
      context.PushState(State::ArrayExpr);
      break;
    }
    case Lex::TokenKind::Package: {
      context.AddLeafNode(NodeKind::PackageExpr, context.Consume());
      if (context.PositionKind() != Lex::TokenKind::Period) {
        CARBON_DIAGNOSTIC(ExpectedPeriodAfterPackage, Error,
                          "expected `.` after `package` expression");
        context.emitter().Emit(*context.position(), ExpectedPeriodAfterPackage);
        state.has_error = true;
      }
      context.PushState(state);
      break;
    }
    case Lex::TokenKind::Core: {
      context.AddLeafNode(NodeKind::CoreNameExpr, context.Consume());
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
    case Lex::TokenKind::Period: {
      // For periods, we look at the next token to form a designator like
      // `.Member` or `.Self`.
      auto period = context.Consume();
      if (context.ConsumeAndAddLeafNodeIf(
              Lex::TokenKind::Identifier,
              NodeKind::IdentifierNameNotBeforeParams)) {
        // OK, `.` identifier.
      } else if (context.ConsumeAndAddLeafNodeIf(
                     Lex::TokenKind::SelfTypeIdentifier,
                     NodeKind::SelfTypeName)) {
        // OK, `.Self`.
      } else {
        CARBON_DIAGNOSTIC(ExpectedIdentifierOrSelfAfterPeriod, Error,
                          "expected identifier or `Self` after `.`");
        context.emitter().Emit(*context.position(),
                               ExpectedIdentifierOrSelfAfterPeriod);
        // Only consume if it is a number or word.
        if (context.PositionKind().is_keyword()) {
          context.AddLeafNode(NodeKind::IdentifierNameNotBeforeParams,
                              context.Consume(), /*has_error=*/true);
        } else if (context.PositionIs(Lex::TokenKind::IntLiteral)) {
          context.AddInvalidParse(context.Consume());
        } else {
          context.AddInvalidParse(*context.position());
          // Indicate the error to the parent state so that it can avoid
          // producing more errors. We only do this on this path where we don't
          // consume the token after the period, where we expect further errors
          // since we likely haven't recovered.
          context.ReturnErrorOnState();
        }
        state.has_error = true;
      }
      context.AddNode(NodeKind::DesignatorExpr, period, state.has_error);
      context.PushState(state);
      break;
    }
    default: {
      // If not already diagnosed in the lexer, diagnose it here.
      if (token_kind != Lex::TokenKind::Error) {
        CARBON_DIAGNOSTIC(ExpectedExpr, Error, "expected expression");
        context.emitter().Emit(*context.position(), ExpectedExpr);
      }

      // Add a node to keep the parse tree balanced.
      context.AddInvalidParse(*context.position());
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
          "parentheses are required to disambiguate operator precedence");
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
                        state.has_error);
        state.state = State::ExprLoopForShortCircuitOperatorAsAnd;
        break;
      case Lex::TokenKind::Or:
        context.AddNode(NodeKind::ShortCircuitOperandOr, state.token,
                        state.has_error);
        state.state = State::ExprLoopForShortCircuitOperatorAsOr;
        break;

      // `where` also needs a virtual parse tree node, and parses its right
      // argument in a mode where it can handle requirement operators like
      // `impls` and `=`.
      case Lex::TokenKind::Where:
        context.AddNode(NodeKind::WhereOperand, state.token, state.has_error);
        context.PushState(state, State::WhereFinish);
        context.PushState(State::RequirementBegin);
        return;

      default:
        state.state = State::ExprLoopForInfixOperator;
        break;
    }

    context.PushState(state);
    context.PushStateForExpr(operator_precedence);
  } else {
    NodeKind node_kind;
    switch (operator_kind) {
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name) \
  case Lex::TokenKind::Name:                          \
    node_kind = NodeKind::PostfixOperator##Name;      \
    break;
#include "toolchain/parse/node_kind.def"

      default:
        CARBON_FATAL("Unexpected token kind for postfix operator: {0}",
                     operator_kind);
    }

    context.AddNode(node_kind, state.token, state.has_error);
    state.has_error = false;
    context.PushState(state);
  }
}

// Adds the operator node and returns the main expression loop.
static auto HandleExprLoopForOperator(Context& context,
                                      Context::StateStackEntry state,
                                      NodeKind node_kind) -> void {
  context.AddNode(node_kind, state.token, state.has_error);
  state.has_error = false;
  context.PushState(state, State::ExprLoop);
}

auto HandleExprLoopForInfixOperator(Context& context) -> void {
  auto state = context.PopState();

  switch (auto token_kind = context.tokens().GetKind(state.token)) {
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name)                           \
  case Lex::TokenKind::Name:                                                  \
    HandleExprLoopForOperator(context, state, NodeKind::InfixOperator##Name); \
    break;
#include "toolchain/parse/node_kind.def"

    default:
      CARBON_FATAL("Unexpected token kind for infix operator: {0}", token_kind);
  }
}

auto HandleExprLoopForPrefixOperator(Context& context) -> void {
  auto state = context.PopState();

  switch (auto token_kind = context.tokens().GetKind(state.token)) {
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name)                           \
  case Lex::TokenKind::Name:                                                   \
    HandleExprLoopForOperator(context, state, NodeKind::PrefixOperator##Name); \
    break;
#include "toolchain/parse/node_kind.def"

    default:
      CARBON_FATAL("Unexpected token kind for prefix operator: {0}",
                   token_kind);
  }
}

auto HandleExprLoopForShortCircuitOperatorAsAnd(Context& context) -> void {
  auto state = context.PopState();

  HandleExprLoopForOperator(context, state, NodeKind::ShortCircuitOperatorAnd);
}

auto HandleExprLoopForShortCircuitOperatorAsOr(Context& context) -> void {
  auto state = context.PopState();

  HandleExprLoopForOperator(context, state, NodeKind::ShortCircuitOperatorOr);
}

auto HandleExprStatementFinish(Context& context) -> void {
  auto state = context.PopState();

  if (auto semi = context.ConsumeIf(Lex::TokenKind::Semi)) {
    context.AddNode(NodeKind::ExprStatement, *semi, state.has_error);
    return;
  }

  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ExpectedExprSemi, Error,
                      "expected `;` after expression statement");
    context.emitter().Emit(*context.position(), ExpectedExprSemi);
  }

  context.AddNode(NodeKind::ExprStatement,
                  context.SkipPastLikelyEnd(state.token), /*has_error=*/true);
}

}  // namespace Carbon::Parse

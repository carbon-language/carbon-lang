// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleStatement(Context& context) -> void {
  context.PopAndDiscardState();

  switch (context.PositionKind()) {
    case Lex::TokenKind::Break: {
      context.PushState(State::StatementBreakFinish);
      context.AddLeafNode(NodeKind::BreakStatementStart, context.Consume());
      break;
    }
    case Lex::TokenKind::Continue: {
      context.PushState(State::StatementContinueFinish);
      context.AddLeafNode(NodeKind::ContinueStatementStart, context.Consume());
      break;
    }
    case Lex::TokenKind::For: {
      context.PushState(State::StatementForFinish);
      context.PushState(State::StatementForHeader);
      context.ConsumeAndDiscard();
      break;
    }
    case Lex::TokenKind::If: {
      context.PushState(State::StatementIf);
      break;
    }
    case Lex::TokenKind::Return: {
      context.PushState(State::StatementReturn);
      break;
    }
    case Lex::TokenKind::Returned: {
      // TODO: Consider handling this as a modifier.
      context.PushState(State::VarAsReturned);
      break;
    }
    case Lex::TokenKind::While: {
      context.PushState(State::StatementWhile);
      break;
    }
    case Lex::TokenKind::Match: {
      context.PushState(State::MatchIntroducer);
      break;
    }
#define CARBON_PARSE_NODE_KIND(Name)
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name) case Lex::TokenKind::Name:
#include "toolchain/parse/node_kind.def"
    case Lex::TokenKind::Adapt:
    case Lex::TokenKind::Alias:
    case Lex::TokenKind::Choice:
    case Lex::TokenKind::Class:
    case Lex::TokenKind::Constraint:
    case Lex::TokenKind::Fn:
    case Lex::TokenKind::Import:
    case Lex::TokenKind::Interface:
    case Lex::TokenKind::Let:
    case Lex::TokenKind::Library:
    case Lex::TokenKind::Namespace:
    // We intentionally don't handle Package here, because `package.` can be
    // used at the start of an expression, and it's not worth disambiguating it.
    case Lex::TokenKind::Var: {
      context.PushState(State::Decl);
      break;
    }
    default: {
      context.PushState(State::ExprStatementFinish);
      context.PushStateForExpr(PrecedenceGroup::ForExprStatement());
      break;
    }
  }
}

// Handles the `;` after a keyword statement.
static auto HandleStatementKeywordFinish(Context& context, NodeKind node_kind)
    -> void {
  auto state = context.PopState();

  auto semi = context.ConsumeIf(Lex::TokenKind::Semi);
  if (!semi) {
    CARBON_DIAGNOSTIC(ExpectedStatementSemi, Error,
                      "`{0}` statements must end with a `;`", Lex::TokenKind);
    context.emitter().Emit(*context.position(), ExpectedStatementSemi,
                           context.tokens().GetKind(state.token));
    state.has_error = true;
    // Recover to the next semicolon if possible.
    semi = context.SkipPastLikelyEnd(state.token);
  }
  context.AddNode(node_kind, *semi, state.has_error);
}

auto HandleStatementBreakFinish(Context& context) -> void {
  HandleStatementKeywordFinish(context, NodeKind::BreakStatement);
}

auto HandleStatementContinueFinish(Context& context) -> void {
  HandleStatementKeywordFinish(context, NodeKind::ContinueStatement);
}

auto HandleStatementForHeader(Context& context) -> void {
  auto state = context.PopState();

  std::optional<Lex::TokenIndex> open_paren =
      context.ConsumeAndAddOpenParen(state.token, NodeKind::ForHeaderStart);
  if (open_paren) {
    state.token = *open_paren;
  }
  state.state = State::StatementForHeaderIn;

  if (context.PositionIs(Lex::TokenKind::Var)) {
    context.PushState(state);
    context.PushState(State::VarAsFor);
    context.AddLeafNode(NodeKind::VariableIntroducer, context.Consume());
  } else {
    CARBON_DIAGNOSTIC(ExpectedVariableDecl, Error,
                      "expected `var` declaration");
    context.emitter().Emit(*context.position(), ExpectedVariableDecl);

    if (auto next_in = context.FindNextOf({Lex::TokenKind::In})) {
      context.SkipTo(*next_in);
      context.ConsumeAndDiscard();
    }
    state.has_error = true;
    context.PushState(state);
  }
}

auto HandleStatementForHeaderIn(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::StatementForHeaderFinish);
  context.PushState(State::Expr);
}

auto HandleStatementForHeaderFinish(Context& context) -> void {
  auto state = context.PopState();

  context.ConsumeAndAddCloseSymbol(state.token, state, NodeKind::ForHeader);

  context.PushState(State::CodeBlock);
}

auto HandleStatementForFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ForStatement, state.token, state.has_error);
}

auto HandleStatementIf(Context& context) -> void {
  context.PopAndDiscardState();

  context.PushState(State::StatementIfConditionFinish);
  context.PushState(State::ParenConditionAsIf);
  context.ConsumeAndDiscard();
}

auto HandleStatementIfConditionFinish(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::StatementIfThenBlockFinish);
  context.PushState(State::CodeBlock);
}

auto HandleStatementIfThenBlockFinish(Context& context) -> void {
  auto state = context.PopState();

  if (context.ConsumeAndAddLeafNodeIf(Lex::TokenKind::Else,
                                      NodeKind::IfStatementElse)) {
    context.PushState(state, State::StatementIfElseBlockFinish);
    // `else if` is permitted as a special case.
    context.PushState(context.PositionIs(Lex::TokenKind::If)
                          ? State::StatementIf
                          : State::CodeBlock);
  } else {
    context.AddNode(NodeKind::IfStatement, state.token, state.has_error);
  }
}

auto HandleStatementIfElseBlockFinish(Context& context) -> void {
  auto state = context.PopState();
  context.AddNode(NodeKind::IfStatement, state.token, state.has_error);
}

auto HandleStatementReturn(Context& context) -> void {
  auto state = context.PopState();
  context.PushState(state, State::StatementReturnFinish);

  context.AddLeafNode(NodeKind::ReturnStatementStart, context.Consume());

  if (auto var_token = context.ConsumeIf(Lex::TokenKind::Var)) {
    // `return var;`
    context.AddLeafNode(NodeKind::ReturnVarModifier, *var_token);
  } else if (!context.PositionIs(Lex::TokenKind::Semi)) {
    // `return <expression>;`
    context.PushState(State::Expr);
  } else {
    // `return;`
  }
}

auto HandleStatementReturnFinish(Context& context) -> void {
  HandleStatementKeywordFinish(context, NodeKind::ReturnStatement);
}

auto HandleStatementScopeLoop(Context& context) -> void {
  // This maintains the current state until we're at the end of the scope.

  auto token_kind = context.PositionKind();
  if (token_kind == Lex::TokenKind::CloseCurlyBrace) {
    auto state = context.PopState();
    if (state.has_error) {
      context.ReturnErrorOnState();
    }
  } else {
    context.PushState(State::Statement);
  }
}

auto HandleStatementWhile(Context& context) -> void {
  context.PopAndDiscardState();

  context.PushState(State::StatementWhileConditionFinish);
  context.PushState(State::ParenConditionAsWhile);
  context.ConsumeAndDiscard();
}

auto HandleStatementWhileConditionFinish(Context& context) -> void {
  auto state = context.PopState();

  context.PushState(state, State::StatementWhileBlockFinish);
  context.PushState(State::CodeBlock);
}

auto HandleStatementWhileBlockFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::WhileStatement, state.token, state.has_error);
}

}  // namespace Carbon::Parse

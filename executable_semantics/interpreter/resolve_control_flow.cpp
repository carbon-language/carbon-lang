// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_control_flow.h"

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/error.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

// Aggregate information about a function being analyzed.
struct FunctionData {
  // The function declaration.
  Nonnull<FunctionDeclaration*> declaration;

  // True if the function has a deduced return type, and we've already seen
  // a `return` statement in its body.
  bool saw_return_in_auto = false;
};

// Resolves control-flow edges such as `Return::function()` and `Break::loop()`
// in the AST rooted at `statement`. `loop` is the innermost loop that
// statically encloses `statement`, or nullopt if there is no such loop.
// `function` carries information about the function body that `statement`
// belongs to, and that information may be updated by this call. `function`
// can be nullopt if `statement` does not belong to a function body, for
// example if it is part of a continuation body instead.
static void ResolveControlFlow(Nonnull<Statement*> statement,
                               std::optional<Nonnull<const Statement*>> loop,
                               std::optional<Nonnull<FunctionData*>> function) {
  switch (statement->kind()) {
    case Statement::Kind::Return: {
      if (!function.has_value()) {
        FATAL_COMPILATION_ERROR(statement->source_loc())
            << "return is not within a function body";
      }
      const ReturnTerm& function_return =
          (*function)->declaration->return_term();
      if (function_return.is_auto()) {
        if ((*function)->saw_return_in_auto) {
          FATAL_COMPILATION_ERROR(statement->source_loc())
              << "Only one return is allowed in a function with an `auto` "
                 "return type.";
        }
        (*function)->saw_return_in_auto = true;
      }
      auto& ret = cast<Return>(*statement);
      ret.set_function((*function)->declaration);
      if (ret.is_omitted_expression() != function_return.is_omitted()) {
        FATAL_COMPILATION_ERROR(ret.source_loc())
            << ret << " should" << (function_return.is_omitted() ? " not" : "")
            << " provide a return value, to match the function's signature.";
      }
      return;
    }
    case Statement::Kind::Break:
      if (!loop.has_value()) {
        FATAL_COMPILATION_ERROR(statement->source_loc())
            << "break is not within a loop body";
      }
      cast<Break>(*statement).set_loop(*loop);
      return;
    case Statement::Kind::Continue:
      if (!loop.has_value()) {
        FATAL_COMPILATION_ERROR(statement->source_loc())
            << "continue is not within a loop body";
      }
      cast<Continue>(*statement).set_loop(*loop);
      return;
    case Statement::Kind::If: {
      auto& if_stmt = cast<If>(*statement);
      ResolveControlFlow(&if_stmt.then_block(), loop, function);
      if (if_stmt.else_block().has_value()) {
        ResolveControlFlow(*if_stmt.else_block(), loop, function);
      }
      return;
    }
    case Statement::Kind::Block: {
      auto& block = cast<Block>(*statement);
      for (auto* block_statement : block.statements()) {
        ResolveControlFlow(block_statement, loop, function);
      }
      return;
    }
    case Statement::Kind::While:
      ResolveControlFlow(&cast<While>(*statement).body(), statement, function);
      return;
    case Statement::Kind::Match: {
      auto& match = cast<Match>(*statement);
      for (Match::Clause& clause : match.clauses()) {
        ResolveControlFlow(&clause.statement(), loop, function);
      }
      return;
    }
    case Statement::Kind::Continuation:
      ResolveControlFlow(&cast<Continuation>(*statement).body(), std::nullopt,
                         std::nullopt);
      return;
    case Statement::Kind::ExpressionStatement:
    case Statement::Kind::Assign:
    case Statement::Kind::VariableDefinition:
    case Statement::Kind::Run:
    case Statement::Kind::Await:
      return;
  }
}

void ResolveControlFlow(AST& ast) {
  for (auto declaration : ast.declarations) {
    if (declaration->kind() != Declaration::Kind::FunctionDeclaration) {
      continue;
    }
    auto& function = cast<FunctionDeclaration>(*declaration);
    if (function.body().has_value()) {
      FunctionData data = {.declaration = &function};
      ResolveControlFlow(*function.body(), std::nullopt, &data);
    }
  }
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_control_flow.h"

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/error.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

// Resolves control-flow edges in the AST rooted at `statement`. `return`
// statements will resolve to `*function`, and `break` and `continue`
// statements will resolve to `*loop`. If either parameter is nullopt, that
// indicates a context where the corresponding statements are not permitted.
static void ResolveControlFlow(
    Nonnull<Statement*> statement,
    std::optional<Nonnull<const FunctionDefinition*>> function,
    std::optional<Nonnull<const Statement*>> loop) {
  switch (statement->kind()) {
    case Statement::Kind::Return:
      if (!function.has_value()) {
        FATAL_COMPILATION_ERROR(statement->source_loc())
            << "return is not within a function body";
      }
      cast<Return>(*statement).set_function(*function);
      return;
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
      ResolveControlFlow(if_stmt.ThenStmt(), function, loop);
      if (if_stmt.ElseStmt().has_value()) {
        ResolveControlFlow(*if_stmt.ElseStmt(), function, loop);
      }
      return;
    }
    case Statement::Kind::Sequence: {
      auto& seq = cast<Sequence>(*statement);
      ResolveControlFlow(seq.Stmt(), function, loop);
      if (seq.Next().has_value()) {
        ResolveControlFlow(*seq.Next(), function, loop);
      }
      return;
    }
    case Statement::Kind::Block: {
      auto& block = cast<Block>(*statement);
      if (block.Stmt().has_value()) {
        ResolveControlFlow(*block.Stmt(), function, loop);
      }
      return;
    }
    case Statement::Kind::While:
      ResolveControlFlow(cast<While>(*statement).Body(), function, statement);
      return;
    case Statement::Kind::Match: {
      auto& match = cast<Match>(*statement);
      for (Match::Clause& clause : match.clauses()) {
        ResolveControlFlow(&clause.statement(), function, loop);
      }
      return;
    }
    case Statement::Kind::Continuation:
      ResolveControlFlow(cast<Continuation>(*statement).Body(), std::nullopt,
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

void ResolveControlFlow(Nonnull<Declaration*> declaration) {
  if (declaration->kind() != Declaration::Kind::FunctionDeclaration) {
    return;
  }
  auto& definition = cast<FunctionDeclaration>(*declaration).definition();
  if (definition.body().has_value()) {
    ResolveControlFlow(*definition.body(), &definition, std::nullopt);
  }
}

}  // namespace Carbon

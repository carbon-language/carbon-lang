// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/resolve_control_flow.h"

#include "explorer/ast/declaration.h"
#include "explorer/ast/return_term.h"
#include "explorer/ast/statement.h"
#include "explorer/base/error_builders.h"
#include "explorer/base/print_as_id.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

// Aggregate information about a function being analyzed.
struct FunctionData {
  // The function declaration.
  Nonnull<CallableDeclaration*> declaration;

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
static auto ResolveControlFlow(Nonnull<TraceStream*> trace_stream,
                               Nonnull<Statement*> statement,
                               std::optional<Nonnull<const Statement*>> loop,
                               std::optional<Nonnull<FunctionData*>> function)
    -> ErrorOr<Success> {
  SetFileContext set_file_ctx(*trace_stream, statement->source_loc());

  switch (statement->kind()) {
    case StatementKind::ReturnVar:
    case StatementKind::ReturnExpression: {
      if (!function.has_value()) {
        return ProgramError(statement->source_loc())
               << "return is not within a function body";
      }
      const ReturnTerm& function_return =
          (*function)->declaration->return_term();
      if (function_return.is_auto()) {
        if ((*function)->saw_return_in_auto) {
          return ProgramError(statement->source_loc())
                 << "Only one return is allowed in a function with an `auto` "
                    "return type.";
        }
        (*function)->saw_return_in_auto = true;
      }
      auto& ret = cast<Return>(*statement);
      ret.set_function((*function)->declaration);
      if (statement->kind() == StatementKind::ReturnVar &&
          function_return.is_omitted()) {
        return ProgramError(statement->source_loc())
               << *statement
               << " should not provide a return value, to match the function's "
                  "signature.";
      }
      if (statement->kind() == StatementKind::ReturnExpression) {
        auto& ret_exp = cast<ReturnExpression>(*statement);
        if (ret_exp.is_omitted_expression() != function_return.is_omitted()) {
          return ProgramError(ret_exp.source_loc())
                 << ret_exp << " should"
                 << (function_return.is_omitted() ? " not" : "")
                 << " provide a return value, to match the function's "
                    "signature.";
        }
      }

      if (trace_stream->is_enabled()) {
        trace_stream->Result()
            << "flow-resolved return statement `" << *statement << "` in `"
            << PrintAsID(*((*function)->declaration)) << "` ("
            << statement->source_loc() << ")\n";
      }

      return Success();
    }
    case StatementKind::Break:
      if (!loop.has_value()) {
        return ProgramError(statement->source_loc())
               << "break is not within a loop body";
      }
      cast<Break>(*statement).set_loop(*loop);

      if (trace_stream->is_enabled()) {
        trace_stream->Result()
            << "flow-resolved break statement `" << *statement << "` for `"
            << PrintAsID(**loop) << "`\n";
      }

      return Success();
    case StatementKind::Continue:
      if (!loop.has_value()) {
        return ProgramError(statement->source_loc())
               << "continue is not within a loop body";
      }
      cast<Continue>(*statement).set_loop(*loop);

      if (trace_stream->is_enabled()) {
        trace_stream->Result()
            << "flow-resolved continue statement `" << *statement << "` in `"
            << PrintAsID(**loop) << "` (" << statement->source_loc() << ")\n";
      }

      return Success();
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*statement);
      CARBON_RETURN_IF_ERROR(ResolveControlFlow(
          trace_stream, &if_stmt.then_block(), loop, function));
      if (if_stmt.else_block().has_value()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(
            trace_stream, *if_stmt.else_block(), loop, function));
      }
      return Success();
    }
    case StatementKind::Block: {
      auto& block = cast<Block>(*statement);
      for (auto* block_statement : block.statements()) {
        CARBON_RETURN_IF_ERROR(
            ResolveControlFlow(trace_stream, block_statement, loop, function));
      }
      return Success();
    }
    case StatementKind::For: {
      CARBON_RETURN_IF_ERROR(ResolveControlFlow(
          trace_stream, &cast<For>(*statement).body(), statement, function));

      if (trace_stream->is_enabled()) {
        trace_stream->Result()
            << "flow-resolved for statement `" << PrintAsID(*statement) << "` ("
            << statement->source_loc() << ")\n";
      }

      return Success();
    }
    case StatementKind::While:
      CARBON_RETURN_IF_ERROR(ResolveControlFlow(
          trace_stream, &cast<While>(*statement).body(), statement, function));

      if (trace_stream->is_enabled()) {
        trace_stream->Result()
            << "flow-resolved while statement `" << PrintAsID(*statement)
            << "` (" << statement->source_loc() << ")\n";
      }

      return Success();
    case StatementKind::Match: {
      auto& match = cast<Match>(*statement);
      for (Match::Clause& clause : match.clauses()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(
            trace_stream, &clause.statement(), loop, function));
      }
      return Success();
    }
    case StatementKind::ExpressionStatement:
    case StatementKind::Assign:
    case StatementKind::IncrementDecrement:
    case StatementKind::VariableDefinition:
      return Success();
  }
}

auto ResolveControlFlow(Nonnull<TraceStream*> trace_stream,
                        Nonnull<Declaration*> declaration) -> ErrorOr<Success> {
  switch (declaration->kind()) {
    case DeclarationKind::DestructorDeclaration:
    case DeclarationKind::FunctionDeclaration: {
      auto& callable = cast<CallableDeclaration>(*declaration);
      if (callable.body().has_value()) {
        FunctionData data = {.declaration = &callable};
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(
            trace_stream, *callable.body(), std::nullopt, &data));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*declaration);
      for (Nonnull<Declaration*> member : class_decl.members()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, member));
      }
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      auto& mixin_decl = cast<MixinDeclaration>(*declaration);
      for (Nonnull<Declaration*> member : mixin_decl.members()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, member));
      }
      break;
    }
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      auto& iface_decl = cast<ConstraintTypeDeclaration>(*declaration);
      for (Nonnull<Declaration*> member : iface_decl.members()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, member));
      }
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*declaration);
      for (Nonnull<Declaration*> member : impl_decl.members()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, member));
      }
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      auto& match_first_decl = cast<MatchFirstDeclaration>(*declaration);
      for (Nonnull<Declaration*> impl : match_first_decl.impl_declarations()) {
        CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, impl));
      }
      break;
    }
    case DeclarationKind::NamespaceDeclaration:
    case DeclarationKind::ChoiceDeclaration:
    case DeclarationKind::VariableDeclaration:
    case DeclarationKind::InterfaceExtendDeclaration:
    case DeclarationKind::InterfaceRequireDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration:
    case DeclarationKind::SelfDeclaration:
    case DeclarationKind::AliasDeclaration:
    case DeclarationKind::MixDeclaration:
    case DeclarationKind::ExtendBaseDeclaration:
      // do nothing
      break;
  }
  return Success();
}

auto ResolveControlFlow(Nonnull<TraceStream*> trace_stream, AST& ast)
    -> ErrorOr<Success> {
  for (auto* declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(ResolveControlFlow(trace_stream, declaration));
  }
  return Success();
}

}  // namespace Carbon

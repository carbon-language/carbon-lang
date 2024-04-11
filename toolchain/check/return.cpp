// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

// Gets the function that lexically encloses the current location.
static auto GetCurrentFunction(Context& context) -> SemIR::Function& {
  CARBON_CHECK(!context.return_scope_stack().empty())
      << "Handling return but not in a function";
  auto function_id = context.insts()
                         .GetAs<SemIR::FunctionDecl>(
                             context.return_scope_stack().back().decl_id)
                         .function_id;
  return context.functions().Get(function_id);
}

// Gets the currently in scope `returned var`, if any, that would be returned
// by a `return var;`.
static auto GetCurrentReturnedVar(Context& context) -> SemIR::InstId {
  CARBON_CHECK(!context.return_scope_stack().empty())
      << "Handling return but not in a function";
  return context.return_scope_stack().back().returned_var;
}

// Produces a note that the given function has no explicit return type.
static auto NoteNoReturnTypeProvided(Context::DiagnosticBuilder& diag,
                                     const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeOmittedNote, Note,
                    "There was no return type provided.");
  diag.Note(function.decl_id, ReturnTypeOmittedNote);
}

// Produces a note describing the return type of the given function.
static auto NoteReturnType(Context::DiagnosticBuilder& diag,
                           const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeHereNote, Note,
                    "Return type of function is `{0}`.", SemIR::TypeId);
  diag.Note(function.return_storage_id, ReturnTypeHereNote,
            function.return_type_id);
}

// Produces a note pointing at the currently in scope `returned var`.
static auto NoteReturnedVar(Context::DiagnosticBuilder& diag,
                            SemIR::InstId returned_var_id) {
  CARBON_DIAGNOSTIC(ReturnedVarHere, Note, "`returned var` was declared here.");
  diag.Note(returned_var_id, ReturnedVarHere);
}

auto CheckReturnedVar(Context& context, Parse::NodeId returned_node,
                      Parse::NodeId name_node, SemIR::NameId name_id,
                      Parse::NodeId type_node, SemIR::TypeId type_id)
    -> SemIR::InstId {
  // A `returned var` requires an explicit return type.
  auto& function = GetCurrentFunction(context);
  if (!function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnedVarWithNoReturnType, Error,
                      "Cannot declare a `returned var` in this function.");
    auto diag =
        context.emitter().Build(returned_node, ReturnedVarWithNoReturnType);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    return SemIR::InstId::BuiltinError;
  }

  // The declared type of the var must match the return type of the function.
  if (function.return_type_id != type_id) {
    CARBON_DIAGNOSTIC(ReturnedVarWrongType, Error,
                      "Type `{0}` of `returned var` does not match "
                      "return type of enclosing function.",
                      SemIR::TypeId);
    auto diag =
        context.emitter().Build(type_node, ReturnedVarWrongType, type_id);
    NoteReturnType(diag, function);
    diag.Emit();
    return SemIR::InstId::BuiltinError;
  }

  // The variable aliases the return slot if there is one. If not, it has its
  // own storage.
  if (function.has_return_slot()) {
    return function.return_storage_id;
  }
  return context.AddInst({name_node, SemIR::VarStorage{type_id, name_id}});
}

auto RegisterReturnedVar(Context& context, SemIR::InstId bind_id) -> void {
  auto existing_id = context.scope_stack().SetReturnedVarOrGetExisting(bind_id);
  if (existing_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnedVarShadowed, Error,
                      "Cannot declare a `returned var` in the scope of "
                      "another `returned var`.");
    auto diag = context.emitter().Build(bind_id, ReturnedVarShadowed);
    NoteReturnedVar(diag, existing_id);
    diag.Emit();
  }
}

auto BuildReturnWithNoExpr(Context& context, Parse::ReturnStatementId node_id)
    -> void {
  const auto& function = GetCurrentFunction(context);

  if (function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnStatementMissingExpr, Error,
                      "Missing return value.");
    auto diag = context.emitter().Build(node_id, ReturnStatementMissingExpr);
    NoteReturnType(diag, function);
    diag.Emit();
  }

  context.AddInst({node_id, SemIR::Return{}});
}

auto BuildReturnWithExpr(Context& context, Parse::ReturnStatementId node_id,
                         SemIR::InstId expr_id) -> void {
  const auto& function = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpr, Error,
        "No return expression should be provided in this context.");
    auto diag = context.emitter().Build(node_id, ReturnStatementDisallowExpr);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "Can only `return var;` in the scope of a `returned var`.");
    auto diag = context.emitter().Build(node_id, ReturnExprWithReturnedVar);
    NoteReturnedVar(diag, returned_var_id);
    diag.Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (function.has_return_slot()) {
    expr_id = Initialize(context, node_id, function.return_storage_id, expr_id);
  } else if (function.return_slot == SemIR::Function::ReturnSlot::Error) {
    // Don't produce a second error complaining the return type is incomplete.
    expr_id = SemIR::InstId::BuiltinError;
  } else {
    expr_id = ConvertToValueOfType(context, node_id, expr_id,
                                   function.return_type_id);
  }

  context.AddInst({node_id, SemIR::ReturnExpr{expr_id}});
}

auto BuildReturnVar(Context& context, Parse::ReturnStatementId node_id)
    -> void {
  const auto& function = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnVarWithNoReturnedVar, Error,
                      "`return var;` with no `returned var` in scope.");
    context.emitter().Emit(node_id, ReturnVarWithNoReturnedVar);
    returned_var_id = SemIR::InstId::BuiltinError;
  }

  if (!function.has_return_slot()) {
    // If we don't have a return slot, we're returning by value. Convert to a
    // value expression.
    returned_var_id = ConvertToValueExpr(context, returned_var_id);
  }

  context.AddInst({node_id, SemIR::ReturnExpr{returned_var_id}});
}

}  // namespace Carbon::Check

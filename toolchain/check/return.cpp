// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/return.h"

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/inst.h"

namespace Carbon::Check {

// Gets the function that lexically encloses the current location.
auto GetCurrentFunctionForReturn(Context& context) -> SemIR::Function& {
  CARBON_CHECK(!context.return_scope_stack().empty(),
               "Handling return but not in a function");
  auto function_id = context.insts()
                         .GetAs<SemIR::FunctionDecl>(
                             context.return_scope_stack().back().decl_id)
                         .function_id;
  return context.functions().Get(function_id);
}

auto GetCurrentReturnSlot(Context& context) -> SemIR::InstId {
  // TODO: this does some unnecessary work to compute non-lexical scopes,
  // so a separate API on ScopeStack could be more efficient.
  auto return_slot_id = context.scope_stack()
                            .LookupInLexicalScopes(SemIR::NameId::ReturnSlot)
                            .first;
  return return_slot_id;
}

// Gets the currently in scope `returned var`, if any, that would be returned
// by a `return var;`.
static auto GetCurrentReturnedVar(Context& context) -> SemIR::InstId {
  CARBON_CHECK(!context.return_scope_stack().empty(),
               "Handling return but not in a function");
  return context.return_scope_stack().back().returned_var;
}

// Produces a note that the given function has no explicit return type.
static auto NoteNoReturnTypeProvided(Context::DiagnosticBuilder& diag,
                                     const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeOmittedNote, Note,
                    "there was no return type provided");
  diag.Note(function.latest_decl_id(), ReturnTypeOmittedNote);
}

// Produces a note describing the return type of the given function, which
// must be a function whose definition is currently being checked.
static auto NoteReturnType(Context& context, Context::DiagnosticBuilder& diag,
                           const SemIR::Function& function) {
  auto out_param_pattern = context.insts().GetAs<SemIR::OutParamPattern>(
      function.return_slot_pattern_id);
  auto return_type_inst_id =
      context.insts()
          .GetAs<SemIR::ReturnSlotPattern>(out_param_pattern.subpattern_id)
          .type_inst_id;
  CARBON_DIAGNOSTIC(ReturnTypeHereNote, Note, "return type of function is {0}",
                    InstIdAsType);
  diag.Note(function.return_slot_pattern_id, ReturnTypeHereNote,
            return_type_inst_id);
}

// Produces a note pointing at the currently in scope `returned var`.
static auto NoteReturnedVar(Context::DiagnosticBuilder& diag,
                            SemIR::InstId returned_var_id) {
  CARBON_DIAGNOSTIC(ReturnedVarHere, Note, "`returned var` was declared here");
  diag.Note(returned_var_id, ReturnedVarHere);
}

auto RegisterReturnedVar(Context& context, Parse::NodeId returned_node,
                         Parse::NodeId type_node, SemIR::TypeId type_id,
                         SemIR::InstId bind_id) -> void {
  auto& function = GetCurrentFunctionForReturn(context);
  auto return_info =
      SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(), function);
  if (!return_info.is_valid()) {
    // We already diagnosed this when we started defining the function.
    return;
  }

  // A `returned var` requires an explicit return type.
  if (!return_info.type_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnedVarWithNoReturnType, Error,
                      "cannot declare a `returned var` in this function");
    auto diag =
        context.emitter().Build(returned_node, ReturnedVarWithNoReturnType);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    return;
  }

  // The declared type of the var must match the return type of the function.
  if (return_info.type_id != type_id) {
    CARBON_DIAGNOSTIC(ReturnedVarWrongType, Error,
                      "type {0} of `returned var` does not match "
                      "return type of enclosing function",
                      SemIR::TypeId);
    auto diag =
        context.emitter().Build(type_node, ReturnedVarWrongType, type_id);
    NoteReturnType(context, diag, function);
    diag.Emit();
  }

  auto existing_id = context.scope_stack().SetReturnedVarOrGetExisting(bind_id);
  if (existing_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnedVarShadowed, Error,
                      "cannot declare a `returned var` in the scope of "
                      "another `returned var`");
    auto diag = context.emitter().Build(bind_id, ReturnedVarShadowed);
    NoteReturnedVar(diag, existing_id);
    diag.Emit();
  }
}

auto BuildReturnWithNoExpr(Context& context, Parse::ReturnStatementId node_id)
    -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());

  if (return_type_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnStatementMissingExpr, Error,
                      "missing return value");
    auto diag = context.emitter().Build(node_id, ReturnStatementMissingExpr);
    NoteReturnType(context, diag, function);
    diag.Emit();
  }

  AddInst<SemIR::Return>(context, node_id, {});
}

auto BuildReturnWithExpr(Context& context, Parse::ReturnStatementId node_id,
                         SemIR::InstId expr_id) -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto returned_var_id = GetCurrentReturnedVar(context);
  auto return_slot_id = SemIR::InstId::None;
  auto return_info =
      SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(), function);

  if (!return_info.type_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpr, Error,
        "no return expression should be provided in this context");
    auto diag = context.emitter().Build(node_id, ReturnStatementDisallowExpr);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    expr_id = SemIR::ErrorInst::SingletonInstId;
  } else if (returned_var_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "can only `return var;` in the scope of a `returned var`");
    auto diag = context.emitter().Build(node_id, ReturnExprWithReturnedVar);
    NoteReturnedVar(diag, returned_var_id);
    diag.Emit();
    expr_id = SemIR::ErrorInst::SingletonInstId;
  } else if (!return_info.is_valid()) {
    // We already diagnosed that the return type is invalid. Don't try to
    // convert to it.
    expr_id = SemIR::ErrorInst::SingletonInstId;
  } else if (return_info.has_return_slot()) {
    return_slot_id = GetCurrentReturnSlot(context);
    CARBON_CHECK(return_slot_id.has_value());
    // Note that this can import a function and invalidate `function`.
    expr_id = Initialize(context, node_id, return_slot_id, expr_id);
  } else {
    expr_id =
        ConvertToValueOfType(context, node_id, expr_id, return_info.type_id);
  }

  AddInst<SemIR::ReturnExpr>(context, node_id,
                             {.expr_id = expr_id, .dest_id = return_slot_id});
}

auto BuildReturnVar(Context& context, Parse::ReturnStatementId node_id)
    -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!returned_var_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnVarWithNoReturnedVar, Error,
                      "`return var;` with no `returned var` in scope");
    context.emitter().Emit(node_id, ReturnVarWithNoReturnedVar);
    returned_var_id = SemIR::ErrorInst::SingletonInstId;
  }

  auto return_slot_id = GetCurrentReturnSlot(context);
  if (!SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(), function)
           .has_return_slot()) {
    // If we don't have a return slot, we're returning by value. Convert to a
    // value expression.
    returned_var_id = ConvertToValueExpr(context, returned_var_id);
    return_slot_id = SemIR::InstId::None;
  }

  AddInst<SemIR::ReturnExpr>(
      context, node_id,
      {.expr_id = returned_var_id, .dest_id = return_slot_id});
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

static auto GetCurrentFunction(Context& context) -> SemIR::Function& {
  CARBON_CHECK(!context.return_scope_stack().empty())
      << "Handling return but not in a function";
  auto function_id = context.insts()
                         .GetAs<SemIR::FunctionDeclaration>(
                             context.return_scope_stack().back().decl_id)
                         .function_id;
  return context.functions().Get(function_id);
}

static auto GetCurrentReturnedVar(Context& context) -> SemIR::InstId {
  CARBON_CHECK(!context.return_scope_stack().empty())
      << "Handling return but not in a function";
  return context.return_scope_stack().back().returned_var;
}

static auto NoteNoReturnTypeProvided(Context& context,
                                     Context::DiagnosticBuilder& diag,
                                     const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeOmittedNote, Note,
                    "There was no return type provided.");
  diag.Note(context.insts().Get(function.declaration_id).parse_node(),
            ReturnTypeOmittedNote);
}

static auto NoteReturnType(Context& context, Context::DiagnosticBuilder& diag,
                           const SemIR::Function& function) {
  // TODO: This is the location of the `fn` keyword. Find the location of the
  // return type.
  auto type_parse_node =
      context.insts().Get(function.declaration_id).parse_node();
  CARBON_DIAGNOSTIC(ReturnTypeHereNote, Note,
                    "Return type of function is `{0}`.", std::string);
  diag.Note(type_parse_node, ReturnTypeHereNote,
            context.sem_ir().StringifyType(function.return_type_id, true));
}

static auto NoteReturnedVar(Context& context, Context::DiagnosticBuilder& diag,
                            SemIR::InstId returned_var_id) {
  CARBON_DIAGNOSTIC(ReturnedVarHere, Note, "`returned var` was declared here.");
  diag.Note(context.insts().Get(returned_var_id).parse_node(), ReturnedVarHere);
}

auto CheckReturnedVar(Context& context, Parse::Node returned_node,
                      Parse::Node name_node, IdentifierId name_id,
                      Parse::Node type_node, SemIR::TypeId type_id)
    -> SemIR::InstId {
  // A `returned var` requires an explicit return type.
  auto& function = GetCurrentFunction(context);
  if (!function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnedVarWithNoReturnType, Error,
                      "Cannot declare a `returned var` in this function.");
    auto diag =
        context.emitter().Build(returned_node, ReturnedVarWithNoReturnType);
    NoteNoReturnTypeProvided(context, diag, function);
    diag.Emit();
    return SemIR::InstId::BuiltinError;
  }

  // The declared type of the var must match the return type of the function.
  if (function.return_type_id != type_id) {
    CARBON_DIAGNOSTIC(ReturnedVarWrongType, Error,
                      "Type `{0}` of `returned var` does not match "
                      "return type of enclosing function.",
                      std::string);
    auto diag =
        context.emitter().Build(type_node, ReturnedVarWrongType,
                                context.sem_ir().StringifyType(type_id, true));
    NoteReturnType(context, diag, function);
    diag.Emit();
    return SemIR::InstId::BuiltinError;
  }

  // The variable aliases the return slot if there is one. If not, it has its
  // own storage.
  if (function.return_slot_id.is_valid()) {
    return function.return_slot_id;
  }
  return context.AddInst(SemIR::VarStorage{name_node, type_id, name_id});
}

auto RegisterReturnedVar(Context& context, SemIR::InstId bind_id) -> void {
  auto existing_id = context.SetReturnedVarOrGetExisting(bind_id);
  if (existing_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnedVarShadowed, Error,
                      "Cannot declare a `returned var` in the scope of "
                      "another `returned var`.");
    auto diag = context.emitter().Build(
        context.insts().Get(bind_id).parse_node(), ReturnedVarShadowed);
    NoteReturnedVar(context, diag, existing_id);
    diag.Emit();
  }
}

auto BuildReturnWithNoExpression(Context& context, Parse::Node parse_node)
    -> void {
  const auto& function = GetCurrentFunction(context);

  if (function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                      "Missing return value.", std::string);
    auto diag = context.emitter().Build(
        parse_node, ReturnStatementMissingExpression,
        context.sem_ir().StringifyType(function.return_type_id));
    NoteReturnType(context, diag, function);
    diag.Emit();
  }

  context.AddInst(SemIR::Return{parse_node});
}

auto BuildReturnWithExpression(Context& context, Parse::Node parse_node,
                               SemIR::InstId expr_id) -> void {
  const auto& function = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpression, Error,
        "No return expression should be provided in this context.");
    auto diag =
        context.emitter().Build(parse_node, ReturnStatementDisallowExpression);
    NoteNoReturnTypeProvided(context, diag, function);
    diag.Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "Can only `return var;` in the scope of a `returned var`.");
    auto diag = context.emitter().Build(parse_node, ReturnExprWithReturnedVar);
    NoteReturnedVar(context, diag, returned_var_id);
    diag.Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (function.return_slot_id.is_valid()) {
    expr_id = Initialize(context, parse_node, function.return_slot_id, expr_id);
  } else {
    expr_id = ConvertToValueOfType(context, parse_node, expr_id,
                                   function.return_type_id);
  }

  context.AddInst(SemIR::ReturnExpression{parse_node, expr_id});
}

auto BuildReturnVar(Context& context, Parse::Node parse_node) -> void {
  const auto& function = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnVarWithNoReturnedVar, Error,
                      "`return var;` with no `returned var` in scope.");
    context.emitter().Emit(parse_node, ReturnVarWithNoReturnedVar);
    returned_var_id = SemIR::InstId::BuiltinError;
  }

  if (!function.return_slot_id.is_valid()) {
    // If we don't have a return slot, we're returning by value. Convert to a
    // value expression.
    returned_var_id = ConvertToValueExpression(context, returned_var_id);
  }

  context.AddInst(SemIR::ReturnExpression{parse_node, returned_var_id});
}

}  // namespace Carbon::Check

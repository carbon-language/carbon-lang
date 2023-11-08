// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
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

auto CheckReturnedVar(Context& context, Parse::Node returned_node,
                      Parse::Node name_node, IdentifierId name_id,
                      Parse::Node type_node, SemIR::TypeId type_id)
    -> SemIR::InstId {
  // A `returned var` requires an explicit return type.
  auto& function = GetCurrentFunction(context);
  if (!function.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnedVarWithNoReturnType, Error,
                      "Cannot declare a `returned var` in function with "
                      "no declared return type.");
    // TODO: Produce ReturnStatementImplicitNote here.
    context.emitter().Emit(returned_node, ReturnedVarWithNoReturnType);
    return SemIR::InstId::BuiltinError;
  }

  // The declared type of the var must match the return type of the function.
  if (function.return_type_id != type_id) {
    CARBON_DIAGNOSTIC(ReturnedVarWrongType, Error,
                      "Type `{0}` of `returned var` does not match "
                      "return type `{1}` of enclosing function.",
                      std::string, std::string);
    // TODO: Note the return type location.
    context.emitter().Emit(
        type_node, ReturnedVarWrongType,
        context.sem_ir().StringifyType(type_id, true),
        context.sem_ir().StringifyType(function.return_type_id, true));
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
    CARBON_DIAGNOSTIC(ReturnedVarHere, Note,
                      "`returned var` was declared here.");
    context.emitter()
        .Build(context.insts().Get(bind_id).parse_node(), ReturnedVarShadowed)
        .Note(context.insts().Get(existing_id).parse_node(), ReturnedVarHere)
        .Emit();
  }
}

auto BuildReturnWithNoExpression(Context& context, Parse::Node parse_node)
    -> void {
  const auto& callable = GetCurrentFunction(context);

  if (callable.return_type_id.is_valid()) {
    // TODO: Add a note pointing at the return type's parse node.
    CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                      "Must return a {0}.", std::string);
    context.emitter()
        .Build(parse_node, ReturnStatementMissingExpression,
               context.sem_ir().StringifyType(callable.return_type_id))
        .Emit();
  }

  context.AddInst(SemIR::Return{parse_node});
}

auto BuildReturnWithExpression(Context& context, Parse::Node parse_node,
                               SemIR::InstId expr_id) -> void {
  const auto& callable = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!callable.return_type_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpression, Error,
        "No return expression should be provided in this context.");
    CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                      "There was no return type provided.");
    context.emitter()
        .Build(parse_node, ReturnStatementDisallowExpression)
        .Note(context.insts().Get(callable.declaration_id).parse_node(),
              ReturnStatementImplicitNote)
        .Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "Can only `return var;` in the scope of a `returned var`.");
    context.emitter()
        .Build(parse_node, ReturnExprWithReturnedVar)
        // TODO: Note the location of the `returned var`.
        //.Note(context.insts().Get(return_scope.returned_var).parse_node(),
        // ReturnedVarHere)
        .Emit();
    expr_id = SemIR::InstId::BuiltinError;
  } else if (callable.return_slot_id.is_valid()) {
    expr_id = Initialize(context, parse_node, callable.return_slot_id, expr_id);
  } else {
    expr_id = ConvertToValueOfType(context, parse_node, expr_id,
                                   callable.return_type_id);
  }

  context.AddInst(SemIR::ReturnExpression{parse_node, expr_id});
}

auto BuildReturnVar(Context& context, Parse::Node parse_node) -> void {
  const auto& callable = GetCurrentFunction(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!returned_var_id.is_valid()) {
    CARBON_DIAGNOSTIC(ReturnVarWithNoReturnedVar, Error,
                      "`return var;` with no `returned var` in scope.");
    context.emitter().Emit(parse_node, ReturnVarWithNoReturnedVar);
    returned_var_id = SemIR::InstId::BuiltinError;
  }

  if (!callable.return_slot_id.is_valid()) {
    // If we don't have a return slot, we're returning by value. Convert to a
    // value expression.
    returned_var_id = ConvertToValueExpression(context, returned_var_id);
  }

  context.AddInst(SemIR::ReturnExpression{parse_node, returned_var_id});
}

}  // namespace Carbon::Check

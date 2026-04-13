// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/return.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Gets the function that lexically encloses the current location.
auto GetCurrentFunctionForReturn(Context& context) -> SemIR::Function& {
  CARBON_CHECK(context.scope_stack().IsInFunctionScope(),
               "Handling return but not in a function");
  auto decl_id = context.scope_stack().GetReturnScopeDeclId();
  auto function_id =
      context.insts().GetAs<SemIR::FunctionDecl>(decl_id).function_id;
  return context.functions().Get(function_id);
}

auto GetReturnedVarParam(Context& context, const SemIR::Function& function)
    -> SemIR::InstId {
  auto return_form_id = function.GetDeclaredReturnForm(context.sem_ir());
  if (auto return_form =
          context.insts().TryGetAsIfValid<SemIR::InitForm>(return_form_id)) {
    auto call_params = context.inst_blocks().Get(function.call_params_id);
    CARBON_CHECK(function.call_param_ranges.return_size() == 1);
    auto return_param_id =
        call_params[function.call_param_ranges.return_begin().index];
    auto return_type_id = context.insts().Get(return_param_id).type_id();
    if (SemIR::InitRepr::ForType(context.sem_ir(), return_type_id)
            .MightBeInPlace()) {
      return return_param_id;
    }
  }
  return SemIR::InstId::None;
}

// Gets the currently in scope `returned var` binding, if any, that would be
// returned by a `return var;`.
static auto GetCurrentReturnedVar(Context& context) -> SemIR::InstId {
  CARBON_CHECK(context.scope_stack().IsInFunctionScope(),
               "Handling return but not in a function");
  return context.scope_stack().GetReturnedVar();
}

// Produces a note that the given function has no explicit return type.
static auto NoteNoReturnTypeProvided(DiagnosticBuilder& diag,
                                     const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeOmittedNote, Note,
                    "there was no return type provided");
  diag.Note(function.latest_decl_id(), ReturnTypeOmittedNote);
}

// Produces a note describing the return type of the given function, which
// must be a function whose definition is currently being checked.
static auto NoteReturnType(DiagnosticBuilder& diag,
                           const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeHereNote, Note, "return type of function is {0}",
                    InstIdAsType);
  diag.Note(function.return_type_inst_id, ReturnTypeHereNote,
            function.return_type_inst_id);
}

// Produces a note pointing at the currently in scope `returned var`.
static auto NoteReturnedVar(DiagnosticBuilder& diag,
                            SemIR::InstId returned_var_id) {
  CARBON_DIAGNOSTIC(ReturnedVarHere, Note, "`returned var` was declared here");
  diag.Note(returned_var_id, ReturnedVarHere);
}

auto RegisterReturnedVar(Context& context, Parse::NodeId returned_node,
                         Parse::NodeId type_node, SemIR::TypeId type_id,
                         SemIR::InstId bind_id, SemIR::NameId name_id) -> void {
  auto& function = GetCurrentFunctionForReturn(context);
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());

  // A `returned var` requires an explicit return type.
  if (!return_type_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnedVarWithNoReturnType, Error,
                      "cannot declare a `returned var` in this function");
    auto diag =
        context.emitter().Build(returned_node, ReturnedVarWithNoReturnType);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    return;
  }

  // The declared type of the var must match the return type of the function.
  if (return_type_id != type_id) {
    CARBON_DIAGNOSTIC(ReturnedVarWrongType, Error,
                      "type {0} of `returned var` does not match "
                      "return type of enclosing function",
                      SemIR::TypeId);
    auto diag =
        context.emitter().Build(type_node, ReturnedVarWrongType, type_id);
    NoteReturnType(diag, function);
    diag.Emit();
  }

  auto form_inst_id = function.GetDeclaredReturnForm(context.sem_ir());
  if (!context.insts().Is<SemIR::InitForm>(form_inst_id)) {
    CARBON_DIAGNOSTIC(ReturnedVarNotInit, Error,
                      "`returned var` declaration in function with "
                      "non-initializing return form");
    auto diag = context.emitter().Build(returned_node, ReturnedVarNotInit);
    CARBON_DIAGNOSTIC(ReturnFormHereNote, Note, "return form declared here");
    diag.Note(function.return_form_inst_id, ReturnFormHereNote);
    diag.Emit();
  }

  auto existing_id =
      context.scope_stack().SetReturnedVarOrGetExisting(bind_id, name_id);
  if (existing_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnedVarShadowed, Error,
                      "cannot declare a `returned var` in the scope of "
                      "another `returned var`");
    auto diag = context.emitter().Build(bind_id, ReturnedVarShadowed);
    NoteReturnedVar(diag, existing_id);
    diag.Emit();
  }
}

auto BuildReturnWithNoExpr(Context& context, SemIR::LocId loc_id) -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());

  if (return_type_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnStatementMissingExpr, Error,
                      "missing return value");
    auto diag = context.emitter().Build(loc_id, ReturnStatementMissingExpr);
    NoteReturnType(diag, function);
    diag.Emit();
  }

  AddReturnCleanupBlock(context, loc_id);
}

auto BuildReturnWithExpr(Context& context, SemIR::LocId loc_id,
                         SemIR::InstId expr_id) -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto returned_var_id = GetCurrentReturnedVar(context);
  auto out_param_id = SemIR::InstId::None;

  auto return_type_id = SemIR::TypeId::None;
  if (function.return_type_inst_id.has_value()) {
    return_type_id =
        context.types().GetTypeIdForTypeInstId(function.return_type_inst_id);
  }
  if (!return_type_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpr, Error,
        "no return expression should be provided in this context");
    auto diag = context.emitter().Build(loc_id, ReturnStatementDisallowExpr);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
    expr_id = SemIR::ErrorInst::InstId;
  } else if (returned_var_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "can only `return var;` in the scope of a `returned var`");
    auto diag = context.emitter().Build(loc_id, ReturnExprWithReturnedVar);
    NoteReturnedVar(diag, returned_var_id);
    diag.Emit();
    expr_id = SemIR::ErrorInst::InstId;
  } else {
    auto return_form =
        context.insts().Get(function.GetDeclaredReturnForm(context.sem_ir()));
    CARBON_KIND_SWITCH(return_form) {
      case CARBON_KIND(SemIR::InitForm _): {
        if (!SemIR::InitRepr::ForType(context.sem_ir(), return_type_id)
                 .is_valid() ||
            return_type_id == SemIR::ErrorInst::TypeId) {
          // We already diagnosed that the return type is invalid.
          // Don't try to convert to it.
          expr_id = SemIR::ErrorInst::InstId;
          break;
        }
        auto call_params = context.inst_blocks().Get(
            GetCurrentFunctionForReturn(context).call_params_id);
        if (function.call_param_ranges.return_size() == 0) {
          out_param_id = SemIR::InstId::None;
          break;
        }
        CARBON_CHECK(function.call_param_ranges.return_size() == 1);
        out_param_id =
            call_params[function.call_param_ranges.return_begin().index];
        CARBON_CHECK(out_param_id.has_value());
        expr_id = Initialize(context, loc_id, out_param_id, expr_id,
                             /*for_return=*/true);
        if (!SemIR::InitRepr::ForType(context.sem_ir(), return_type_id)
                 .MightBeInPlace()) {
          out_param_id = SemIR::InstId::None;
        }
        break;
      }
      case CARBON_KIND(SemIR::RefForm ref_form): {
        expr_id = Convert(
            context, loc_id, expr_id,
            ConversionTarget{.kind = ConversionTarget::DurableRef,
                             .type_id = context.types().GetTypeIdForTypeInstId(
                                 ref_form.type_component_inst_id)});
        break;
      }
      case CARBON_KIND(SemIR::ErrorInst _): {
        expr_id = SemIR::ErrorInst::InstId;
        break;
      }
      default:
        CARBON_FATAL("Unexpected inst kind: {0}", return_form);
    }
  }
  AddReturnCleanupBlockWithExpr(context, loc_id,
                                {.expr_id = expr_id, .dest_id = out_param_id});
}

auto BuildReturnVar(Context& context, Parse::ReturnStatementId node_id)
    -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (!returned_var_id.has_value()) {
    CARBON_DIAGNOSTIC(ReturnVarWithNoReturnedVar, Error,
                      "`return var;` with no `returned var` in scope");
    context.emitter().Emit(node_id, ReturnVarWithNoReturnedVar);
    returned_var_id = SemIR::ErrorInst::InstId;
  }

  auto return_param_id = GetReturnedVarParam(context, function);
  if (!return_param_id.has_value()) {
    // If we don't have a return slot, we're returning by value. Convert to a
    // value expression.
    returned_var_id = ConvertToValueExpr(context, returned_var_id);
  }

  AddReturnCleanupBlockWithExpr(
      context, node_id,
      {.expr_id = returned_var_id, .dest_id = return_param_id});
}

}  // namespace Carbon::Check

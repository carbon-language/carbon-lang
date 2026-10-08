// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/return.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/function.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Notes the location from which a function's return type was deduced.
CARBON_DIAGNOSTIC(ReturnTypeDeducedHere, Note,
                  "return type deduced as {0} here", SemIR::TypeId);

// Gets the ID of the function that lexically encloses the current location.
static auto GetCurrentFunctionIdForReturn(Context& context)
    -> SemIR::FunctionId {
  CARBON_CHECK(context.scope_stack().IsInFunctionScope(),
               "Handling return but not in a function");
  auto decl_id = context.scope_stack().GetReturnScopeDeclId();
  return context.insts().GetAs<SemIR::FunctionDecl>(decl_id).function_id;
}

// Gets the function that lexically encloses the current location.
auto GetCurrentFunctionForReturn(Context& context) -> SemIR::Function& {
  return context.functions().Get(GetCurrentFunctionIdForReturn(context));
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

// Produces a note that the return type of the given function is deduced, for a
// function whose return type has not been deduced yet.
static auto NoteReturnTypeIsDeduced(DiagnosticBuilder& diag,
                                    const SemIR::Function& function) {
  CARBON_DIAGNOSTIC(ReturnTypeIsDeducedNote, Note,
                    "return type of function is deduced");
  diag.Note(function.latest_decl_id(), ReturnTypeIsDeducedNote);
}

// Produces a note pointing at the currently in scope `returned var`.
static auto NoteReturnedVar(DiagnosticBuilder& diag,
                            SemIR::InstId returned_var_id) {
  CARBON_DIAGNOSTIC(ReturnedVarHere, Note, "`returned var` was declared here");
  diag.Note(returned_var_id, ReturnedVarHere);
}

namespace {
// The result of converting the operand of a `return` statement to the return
// form of the function.
struct ConvertedReturnExpr {
  // The converted expression.
  SemIR::InstId expr_id;
  // The return slot that the converted expression initializes in place, if
  // any.
  SemIR::InstId dest_id = SemIR::InstId::None;
};
}  // namespace

// Converts the operand of a `return` statement to the declared return form of
// `function`, which must have a declared or deduced return type.
static auto ConvertReturnExpr(Context& context, SemIR::LocId loc_id,
                              const SemIR::Function& function,
                              SemIR::InstId expr_id) -> ConvertedReturnExpr {
  auto return_type_id =
      context.types().GetTypeIdForTypeInstId(function.return_type_inst_id);
  auto return_form_id = function.GetDeclaredReturnForm(context.sem_ir());
  auto return_form = context.insts().Get(return_form_id);
  CARBON_KIND_SWITCH(return_form) {
    case CARBON_KIND(SemIR::InitForm _): {
      if (!SemIR::InitRepr::ForType(context.sem_ir(), return_type_id)
               .is_valid() ||
          return_type_id == SemIR::ErrorInst::TypeId) {
        // We already diagnosed that the return type is invalid.
        // Don't try to convert to it.
        return {.expr_id = SemIR::ErrorInst::InstId};
      }
      if (function.call_param_ranges.return_size() == 0) {
        return {.expr_id = expr_id};
      }
      CARBON_CHECK(function.call_param_ranges.return_size() == 1);
      auto call_params = context.inst_blocks().Get(function.call_params_id);
      auto out_param_id =
          call_params[function.call_param_ranges.return_begin().index];
      CARBON_CHECK(out_param_id.has_value());
      expr_id = InitializeExisting(context, loc_id, out_param_id, expr_id,
                                   /*for_return=*/true);
      if (!SemIR::InitRepr::ForType(context.sem_ir(), return_type_id)
               .MightBeInPlace()) {
        out_param_id = SemIR::InstId::None;
      }
      return {.expr_id = expr_id, .dest_id = out_param_id};
    }
    case CARBON_KIND(SemIR::RefForm ref_form): {
      return {.expr_id =
                  Convert(context, loc_id, expr_id,
                          ConversionTarget{
                              .kind = ConversionTarget::DurableRef,
                              .type_id = context.types().GetTypeIdForTypeInstId(
                                  ref_form.type_component_inst_id)})};
    }
    case CARBON_KIND(SemIR::ValueForm value_form): {
      return {.expr_id =
                  Convert(context, loc_id, expr_id,
                          ConversionTarget{
                              .kind = ConversionTarget::Value,
                              .type_id = context.types().GetTypeIdForTypeInstId(
                                  value_form.type_component_inst_id)})};
    }
    case CARBON_KIND(SemIR::ErrorInst _): {
      return {.expr_id = SemIR::ErrorInst::InstId};
    }
    case CARBON_KIND(SemIR::SymbolicBinding _): {
      auto expr_form_info = SemIR::GetFormInfo(context.sem_ir(), expr_id);
      if (expr_form_info.kind != SemIR::FormInfo::Dependent) {
        context.TODO(loc_id,
                     "support nontrivial conversions between symbolic forms");
        return {.expr_id = SemIR::ErrorInst::InstId};
      }
      auto expr_form_const_id = SemIR::GetConstantValueInSpecific(
          context.sem_ir(), SemIR::SpecificId::None,
          expr_form_info.form_inst_id);
      auto declared_form_const_id = SemIR::GetConstantValueInSpecific(
          context.sem_ir(), SemIR::SpecificId::None, return_form_id);
      if (expr_form_const_id != declared_form_const_id) {
        context.TODO(loc_id,
                     "support nontrivial conversions between symbolic forms");
        return {.expr_id = SemIR::ErrorInst::InstId};
      }
      // expr_id's form is identical to the expected form, so we can use it
      // directly.
      return {.expr_id = expr_id};
    }
    default:
      CARBON_FATAL("Unexpected inst kind: {0}", return_form);
  }
}

// Builds a `return` statement in a function whose return type has not been
// deduced yet. Control flow and cleanups are built now, but the conversion of
// the returned expression to the return type is deferred until the return type
// is known; see `CompletePendingReturns`.
static auto BuildPendingReturn(Context& context, SemIR::LocId loc_id,
                               SemIR::InstId expr_id) -> void {
  // The conversion will be spliced in here. Until then, this placeholder is an
  // identity conversion. Because it precedes the cleanups, the return value is
  // initialized before any local variables are destroyed.
  auto splice_id = AddPlaceholderInst(
      context, loc_id,
      SemIR::SpliceBlock{.type_id = context.insts().Get(expr_id).type_id(),
                         .block_id = SemIR::InstBlockId::Empty,
                         .result_id = expr_id});
  auto return_id = AddReturnInstWithCleanups(
      context, loc_id,
      SemIR::ReturnExpr{.expr_id = splice_id, .dest_id = SemIR::InstId::None});
  context.scope_stack().AddPendingReturn(
      {.expr_id = expr_id, .splice_id = splice_id, .return_id = return_id});
}

// Completes the `return` statements built by `BuildPendingReturn` in the
// current function, once its return type has been deduced.
static auto CompletePendingReturns(Context& context,
                                   SemIR::FunctionId function_id) -> void {
  auto pending_returns = context.scope_stack().TakePendingReturns();
  if (pending_returns.empty()) {
    return;
  }

  const auto& function = context.functions().Get(function_id);

  // Any problems converting to the return type are probably due to the choice
  // of return type, so point out where it came from.
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](DiagnosticBuilder& builder) {
        builder.Note(function.return_type_inst_id, ReturnTypeDeducedHere,
                     context.types().GetTypeIdForTypeInstId(
                         function.return_type_inst_id));
      });

  for (auto [expr_id, splice_id, return_id] : pending_returns) {
    auto loc_id = SemIR::LocId(return_id);

    // Convert in a separate block that we splice in place of the placeholder,
    // with its own cleanup scope for any temporaries.
    context.inst_block_stack().Push();
    context.scope_stack().PushForSameRegion(
        ScopeStack::CleanupScopeKind::Owned);
    auto converted = ConvertReturnExpr(context, loc_id, function, expr_id);
    AddAndDiscardScopeCleanups(context);
    context.scope_stack().Pop();
    auto block_id = context.inst_block_stack().Pop();

    ReplaceInstBeforeConstantUse(
        context, splice_id,
        SemIR::SpliceBlock{
            .type_id = context.insts().Get(converted.expr_id).type_id(),
            .block_id = block_id,
            .result_id = converted.expr_id});
    if (converted.dest_id.has_value()) {
      // The `return` statement has no type or constant value, and nothing
      // refers to it, so it's safe to replace it.
      auto return_inst = context.insts().GetAs<SemIR::ReturnExpr>(return_id);
      return_inst.dest_id = converted.dest_id;
      ReplaceInstBeforeConstantUse(context, return_id, return_inst);
    }
  }
}

auto RegisterReturnedVar(Context& context, Parse::NodeId returned_node,
                         Parse::NodeId type_node, SemIR::TypeId type_id,
                         SemIR::InstId bind_id, SemIR::NameId name_id) -> void {
  auto function_id = GetCurrentFunctionIdForReturn(context);
  if (context.functions().Get(function_id).has_undeduced_return_type()) {
    // In a function with a deduced return type, the first `returned var`
    // determines the return type, and any earlier `return`s are converted to
    // it.
    SetDeducedReturnType(context, function_id, type_node, type_id);
    CompletePendingReturns(context, function_id);
  }

  auto& function = context.functions().Get(function_id);
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
  CARBON_DIAGNOSTIC(ReturnStatementMissingExpr, Error, "missing return value");

  // `-> auto` requires a returned value, just like an explicit return type.
  if (function.has_undeduced_return_type()) {
    auto diag = context.emitter().Build(loc_id, ReturnStatementMissingExpr);
    NoteReturnTypeIsDeduced(diag, function);
    diag.Emit();
    // Treat this as returning an erroneous value, which doesn't participate in
    // return type deduction.
    BuildPendingReturn(context, loc_id, SemIR::ErrorInst::InstId);
    return;
  }

  if (function.GetDeclaredReturnType(context.sem_ir()).has_value()) {
    auto diag = context.emitter().Build(loc_id, ReturnStatementMissingExpr);
    NoteReturnType(diag, function);
    diag.Emit();
  }

  AddReturnInstWithCleanups(context, loc_id);
}

auto BuildReturnWithExpr(Context& context, SemIR::LocId loc_id,
                         SemIR::InstId expr_id) -> void {
  const auto& function = GetCurrentFunctionForReturn(context);
  auto returned_var_id = GetCurrentReturnedVar(context);

  if (function.has_undeduced_return_type()) {
    // Declaring a `returned var` would have deduced the return type.
    CARBON_CHECK(!returned_var_id.has_value());
    BuildPendingReturn(context, loc_id, expr_id);
    return;
  }

  auto return_type_id = SemIR::TypeId::None;
  if (function.return_type_inst_id.has_value()) {
    return_type_id =
        context.types().GetTypeIdForTypeInstId(function.return_type_inst_id);
  }
  auto converted = ConvertedReturnExpr{.expr_id = SemIR::ErrorInst::InstId};
  if (!return_type_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnStatementDisallowExpr, Error,
        "no return expression should be provided in this context");
    auto diag = context.emitter().Build(loc_id, ReturnStatementDisallowExpr);
    NoteNoReturnTypeProvided(diag, function);
    diag.Emit();
  } else if (returned_var_id.has_value()) {
    CARBON_DIAGNOSTIC(
        ReturnExprWithReturnedVar, Error,
        "can only `return var;` in the scope of a `returned var`");
    auto diag = context.emitter().Build(loc_id, ReturnExprWithReturnedVar);
    NoteReturnedVar(diag, returned_var_id);
    diag.Emit();
  } else {
    converted = ConvertReturnExpr(context, loc_id, function, expr_id);
  }
  AddReturnInstWithCleanups(
      context, loc_id,
      {.expr_id = converted.expr_id, .dest_id = converted.dest_id});
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

  if (function.has_undeduced_return_type()) {
    // Declaring a `returned var` would have deduced the return type, so we
    // diagnosed above. Treat this as returning an erroneous value.
    BuildPendingReturn(context, node_id, returned_var_id);
    return;
  }

  auto return_param_id = GetReturnedVarParam(context, function);

  // Convert to a value expression in case the return logic needs a value, and
  // to indicate that this was a `return var`, not a reference return.
  returned_var_id = ConvertToValueExpr(context, returned_var_id);

  AddReturnInstWithCleanups(
      context, node_id,
      {.expr_id = returned_var_id, .dest_id = return_param_id});
}

auto DeduceReturnTypeAtEndOfBody(Context& context,
                                 SemIR::FunctionId function_id) -> bool {
  const auto& function = context.functions().Get(function_id);
  CARBON_CHECK(function.has_undeduced_return_type());
  auto pending_returns = context.scope_stack().PeekPendingReturns();
  auto decl_loc_id = SemIR::LocId(function.latest_decl_id());

  if (pending_returns.empty()) {
    CARBON_DIAGNOSTIC(DeducedReturnTypeWithoutReturn, Error,
                      "no `return` in function with deduced return type");
    context.emitter().Emit(decl_loc_id, DeducedReturnTypeWithoutReturn);
    SetDeducedReturnType(context, function_id, decl_loc_id,
                         SemIR::ErrorInst::TypeId);
    return false;
  }

  // The return type is the type of the returned expressions. Erroneous returned
  // expressions are ignored, to avoid follow-on diagnostics.
  // TODO: Use the common type of the returned expressions.
  auto first_expr_id = SemIR::InstId::None;
  auto return_type_id = SemIR::ErrorInst::TypeId;
  for (const auto& pending : pending_returns) {
    auto type_id = context.insts().Get(pending.expr_id).type_id();
    if (type_id == SemIR::ErrorInst::TypeId) {
      continue;
    }
    if (!first_expr_id.has_value()) {
      first_expr_id = pending.expr_id;
      return_type_id = type_id;
    } else if (type_id != return_type_id) {
      CARBON_DIAGNOSTIC(DeducedReturnTypeMismatch, Error,
                        "`return` of type {0} does not match earlier `return` "
                        "of type {1}",
                        TypeOfInstId, TypeOfInstId);
      context.emitter()
          .Build(pending.expr_id, DeducedReturnTypeMismatch, pending.expr_id,
                 first_expr_id)
          .Note(first_expr_id, ReturnTypeDeducedHere, return_type_id)
          .Emit();
      return_type_id = SemIR::ErrorInst::TypeId;
      break;
    }
  }

  SetDeducedReturnType(
      context, function_id,
      first_expr_id.has_value() ? SemIR::LocId(first_expr_id) : decl_loc_id,
      return_type_id);
  CompletePendingReturns(context, function_id);
  return true;
}

}  // namespace Carbon::Check

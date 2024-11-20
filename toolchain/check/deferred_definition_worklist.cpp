// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/deferred_definition_worklist.h"

#include "common/variant_helpers.h"
#include "common/vlog.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

static constexpr llvm::StringLiteral VlogPrefix = "DeferredDefinitionWorklist ";

DeferredDefinitionWorklist::DeferredDefinitionWorklist(
    llvm::raw_ostream* vlog_stream)
    : vlog_stream_(vlog_stream) {
  // See declaration of `worklist_`.
  worklist_.reserve(64);
}

auto DeferredDefinitionWorklist::SuspendFunctionAndPush(
    Context& context, Parse::DeferredDefinitionIndex index,
    Parse::FunctionDefinitionStartId node_id) -> void {
  // TODO: Investigate factoring out `HandleFunctionDefinitionSuspend` to make
  // `DeferredDefinitionWorklist` reusable.
  worklist_.push_back(CheckSkippedDefinition{
      index, HandleFunctionDefinitionSuspend(context, node_id)});
  CARBON_VLOG("{0}Push CheckSkippedDefinition {1}\n", VlogPrefix, index.index);
}

auto DeferredDefinitionWorklist::PushEnterDeferredDefinitionScope(
    Context& context) -> void {
  bool nested = !entered_scopes_.empty() &&
                entered_scopes_.back().scope_index ==
                    context.decl_name_stack().PeekInitialScopeIndex();
  entered_scopes_.push_back({.worklist_start_index = worklist_.size(),
                             .scope_index = context.scope_stack().PeekIndex()});
  worklist_.push_back(EnterDeferredDefinitionScope{
      .suspended_name = std::nullopt, .in_deferred_definition_scope = nested});
  CARBON_VLOG("{0}Push EnterDeferredDefinitionScope {1}\n", VlogPrefix,
              nested ? "(nested)" : "(non-nested)");
}

auto DeferredDefinitionWorklist::SuspendFinishedScopeAndPush(Context& context)
    -> bool {
  auto start_index = entered_scopes_.pop_back_val().worklist_start_index;

  // If we've not found any deferred definitions in this scope, clean up the
  // stack.
  if (start_index == worklist_.size() - 1) {
    context.decl_name_stack().PopScope();
    worklist_.pop_back();
    CARBON_VLOG("{0}Pop EnterDeferredDefinitionScope (empty)\n", VlogPrefix);
    return false;
  }

  // If we're finishing a nested deferred definition scope, keep track of that
  // but don't type-check deferred definitions now.
  auto& enter_scope = get<EnterDeferredDefinitionScope>(worklist_[start_index]);
  if (enter_scope.in_deferred_definition_scope) {
    // This is a nested deferred definition scope. Suspend the inner scope so we
    // can restore it when we come to type-check the deferred definitions.
    enter_scope.suspended_name = context.decl_name_stack().Suspend();

    // Enqueue a task to leave the nested scope.
    worklist_.push_back(
        LeaveDeferredDefinitionScope{.in_deferred_definition_scope = true});
    CARBON_VLOG("{0}Push LeaveDeferredDefinitionScope (nested)\n", VlogPrefix);
    return false;
  }

  // We're at the end of a non-nested deferred definition scope. Prepare to
  // start checking deferred definitions. Enqueue a task to leave this outer
  // scope and end checking deferred definitions.
  worklist_.push_back(
      LeaveDeferredDefinitionScope{.in_deferred_definition_scope = false});
  CARBON_VLOG("{0}Push LeaveDeferredDefinitionScope (non-nested)\n",
              VlogPrefix);

  // We'll process the worklist in reverse index order, so reverse the part of
  // it we're about to execute so we run our tasks in the order in which they
  // were pushed.
  std::reverse(worklist_.begin() + start_index, worklist_.end());

  // Pop the `EnterDeferredDefinitionScope` that's now on the end of the
  // worklist. We stay in that scope rather than suspending then immediately
  // resuming it.
  CARBON_CHECK(
      holds_alternative<EnterDeferredDefinitionScope>(worklist_.back()),
      "Unexpected task in worklist.");
  worklist_.pop_back();
  CARBON_VLOG("{0}Handle EnterDeferredDefinitionScope (non-nested)\n",
              VlogPrefix);
  return true;
}

auto DeferredDefinitionWorklist::Pop() -> Task {
  if (vlog_stream_) {
    VariantMatch(
        worklist_.back(),
        [&](CheckSkippedDefinition& definition) {
          CARBON_VLOG("{0}Handle CheckSkippedDefinition {1}\n", VlogPrefix,
                      definition.definition_index.index);
        },
        [&](EnterDeferredDefinitionScope& enter) {
          CARBON_CHECK(enter.in_deferred_definition_scope);
          CARBON_VLOG("{0}Handle EnterDeferredDefinitionScope (nested)\n",
                      VlogPrefix);
        },
        [&](LeaveDeferredDefinitionScope& leave) {
          bool nested = leave.in_deferred_definition_scope;
          CARBON_VLOG("{0}Handle LeaveDeferredDefinitionScope {1}\n",
                      VlogPrefix, nested ? "(nested)" : "(non-nested)");
        });
  }

  return worklist_.pop_back_val();
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/deferred_definition_worklist.h"

#include <algorithm>
#include <optional>
#include <variant>

#include "common/emplace_by_calling.h"
#include "common/vlog.h"
#include "toolchain/base/kind_switch.h"
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
  worklist_.emplace_back(EmplaceByCalling([&] {
    return CheckSkippedDefinition{
        index, HandleFunctionDefinitionSuspend(context, node_id)};
  }));
  CARBON_VLOG("{0}Push CheckSkippedDefinition {1}\n", VlogPrefix, index.index);
}

auto DeferredDefinitionWorklist::PushEnterDeferredDefinitionScope(
    Context& context) -> bool {
  bool nested = !entered_scopes_.empty() &&
                entered_scopes_.back().scope_index ==
                    context.decl_name_stack().PeekInitialScopeIndex();
  entered_scopes_.push_back({.nested = nested,
                             .worklist_start_index = worklist_.size(),
                             .scope_index = context.scope_stack().PeekIndex()});
  if (nested) {
    worklist_.emplace_back(EmplaceByCalling([&] {
      return EnterNestedDeferredDefinitionScope{.suspended_name = std::nullopt};
    }));
    CARBON_VLOG("{0}Push EnterDeferredDefinitionScope (nested)\n", VlogPrefix);
  } else {
    // Don't push a task to re-enter a non-nested scope. Instead,
    // SuspendFinishedScopeAndPush will remain in the scope when executing the
    // worklist tasks.
    CARBON_VLOG("{0}Entered non-nested deferred definition scope\n",
                VlogPrefix);
  }
  return !nested;
}

auto DeferredDefinitionWorklist::SuspendFinishedScopeAndPush(Context& context)
    -> FinishedScopeKind {
  auto [nested, start_index, _] = entered_scopes_.pop_back_val();

  // If we've not found any tasks to perform in this scope, clean up the stack.
  // For non-nested scope, there will be no tasks on the worklist for this scope
  // in this case; for a nested scope, there will just be a task to re-enter the
  // nested scope.
  if (!nested && start_index == worklist_.size()) {
    context.decl_name_stack().PopScope();
    CARBON_VLOG("{0}Left non-nested empty deferred definition scope\n",
                VlogPrefix);
    return FinishedScopeKind::NonNestedEmpty;
  }
  if (nested && start_index == worklist_.size() - 1) {
    CARBON_CHECK(std::holds_alternative<EnterNestedDeferredDefinitionScope>(
        worklist_.back()));
    worklist_.pop_back();
    context.decl_name_stack().PopScope();
    CARBON_VLOG("{0}Pop EnterNestedDeferredDefinitionScope (empty)\n",
                VlogPrefix);
    return FinishedScopeKind::Nested;
  }

  // If we're finishing a nested deferred definition scope, keep track of that
  // but don't type-check deferred definitions now.
  if (nested) {
    auto& enter_scope =
        get<EnterNestedDeferredDefinitionScope>(worklist_[start_index]);
    // This is a nested deferred definition scope. Suspend the inner scope so we
    // can restore it when we come to type-check the deferred definitions.
    enter_scope.suspended_name.emplace(
        EmplaceByCalling([&] { return context.decl_name_stack().Suspend(); }));

    // Enqueue a task to leave the nested scope.
    worklist_.emplace_back(LeaveNestedDeferredDefinitionScope{});
    CARBON_VLOG("{0}Push LeaveNestedDeferredDefinitionScope\n", VlogPrefix);
    return FinishedScopeKind::Nested;
  }

  // We're at the end of a non-nested deferred definition scope. Start checking
  // deferred definitions.
  CARBON_VLOG("{0}Starting deferred definition processing\n", VlogPrefix);
  return FinishedScopeKind::NonNestedWithWork;
}

}  // namespace Carbon::Check

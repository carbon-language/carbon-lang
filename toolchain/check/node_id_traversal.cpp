// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/node_id_traversal.h"

#include <optional>
#include <utility>
#include <variant>

#include "toolchain/check/deferred_definition_worklist.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/thunk.h"

namespace Carbon::Check {

NodeIdTraversal::NodeIdTraversal(Context* context,
                                 llvm::raw_ostream* vlog_stream)
    : context_(context),
      next_deferred_definition_(&context->parse_tree()),
      worklist_(vlog_stream) {
  auto range = context->parse_tree().postorder();
  chunks_.push_back({.it = range.begin(),
                     .end = range.end(),
                     .next_definition = Parse::DeferredDefinitionIndex::None});
}

auto NodeIdTraversal::Next() -> std::optional<Parse::NodeId> {
  while (true) {
    // If we're checking deferred definitions, find the next definition we
    // should check, restore its suspended state, and add a corresponding
    // `Chunk` to the top of the chunk list.
    if (chunks_.back().checking_deferred_definitions) {
      if (chunks_.back().next_worklist_index < worklist_.size()) {
        std::visit([&](auto& task) { PerformTask(std::move(task)); },
                   worklist_[chunks_.back().next_worklist_index++]);
        continue;
      }

      // Worklist is empty: discard the worklist items associated with this
      // chunk, and leave the scope.
      worklist_.truncate(chunks_.back().first_worklist_index);
      // We reach here when
      // `DeferredDefinitionScope::SuspendFinishedScopeAndPush` returns
      // `NonNestedWithWork`. In this case it's our responsibility to pop the
      // scope left behind by the `Handle*Definition` function for the
      // non-nested definition.
      context_->decl_name_stack().PopScope();
      chunks_.back().checking_deferred_definitions = false;
    }

    // If we're not checking deferred definitions, produce the next parse node
    // for this chunk. If we've run out of parse nodes, we're done with this
    // chunk of the parse tree.
    if (chunks_.back().it == chunks_.back().end) {
      auto old_chunk = chunks_.pop_back_val();

      // If we're out of chunks, then we're done entirely.
      if (chunks_.empty()) {
        worklist_.VerifyEmpty();
        return std::nullopt;
      }

      next_deferred_definition_.SkipTo(old_chunk.next_definition);
      continue;
    }

    auto node_id = *chunks_.back().it;

    // If we've reached the start of a deferred definition, skip to the end of
    // it, and track that we need to check it later.
    if (node_id == next_deferred_definition_.start_id()) {
      const auto& definition_info =
          context_->parse_tree().deferred_definitions().Get(
              next_deferred_definition_.index());
      worklist_.SuspendFunctionAndPush(*context_,
                                       next_deferred_definition_.index(),
                                       definition_info.start_id);

      // Continue type-checking the parse tree after the end of the definition.
      chunks_.back().it =
          Parse::Tree::PostorderIterator(definition_info.definition_id) + 1;
      next_deferred_definition_.SkipTo(definition_info.next_definition_index);
      continue;
    }

    ++chunks_.back().it;
    return node_id;
  }
}

// Determines whether this node kind is the start of a deferred definition
// scope.
static auto IsStartOfDeferredDefinitionScope(Parse::NodeKind kind) -> bool {
  switch (kind) {
    case Parse::NodeKind::ClassDefinitionStart:
    case Parse::NodeKind::ImplDefinitionStart:
    case Parse::NodeKind::InterfaceDefinitionStart:
    case Parse::NodeKind::NamedConstraintDefinitionStart:
      // TODO: Mixins.
      return true;
    default:
      return false;
  }
}

// Determines whether this node kind is the end of a deferred definition scope.
static auto IsEndOfDeferredDefinitionScope(Parse::NodeKind kind) -> bool {
  switch (kind) {
    case Parse::NodeKind::ClassDefinition:
    case Parse::NodeKind::ImplDefinition:
    case Parse::NodeKind::InterfaceDefinition:
    case Parse::NodeKind::NamedConstraintDefinition:
      // TODO: Mixins.
      return true;
    default:
      return false;
  }
}

// TODO: Investigate factoring out `IsStartOfDeferredDefinitionScope` and
// `IsEndOfDeferredDefinitionScope` in order to make `NodeIdTraversal`
// reusable.
auto NodeIdTraversal::Handle(Parse::NodeKind parse_kind) -> void {
  // When we reach the start of a deferred definition scope, add a task to the
  // worklist to check future skipped definitions in the new context.
  if (IsStartOfDeferredDefinitionScope(parse_kind)) {
    if (worklist_.PushEnterDeferredDefinitionScope(*context_)) {
      // Track that we're within a new non-nested deferred definition scope.
      context_->deferred_definition_scope_stack().Push();
    }
  }

  // When we reach the end of a deferred definition scope, add a task to the
  // worklist to leave the scope. If this is not a nested scope, start
  // checking the deferred definitions now.
  if (IsEndOfDeferredDefinitionScope(parse_kind)) {
    auto scope_kind = worklist_.SuspendFinishedScopeAndPush(*context_);

    // At the end of a non-nested scope, define any pending thunks and clean up
    // the stack.
    if (scope_kind != DeferredDefinitionWorklist::FinishedScopeKind::Nested) {
      for (auto& thunk :
           context_->deferred_definition_scope_stack().PeekPendingThunks()) {
        BuildThunkDefinition(*context_, std::move(thunk));
      }
      context_->deferred_definition_scope_stack().Pop();
    }

    // If we have function definitions in this scope, process them next.
    if (scope_kind ==
        DeferredDefinitionWorklist::FinishedScopeKind::NonNestedWithWork) {
      chunks_.back().checking_deferred_definitions = true;
      chunks_.back().next_worklist_index = chunks_.back().first_worklist_index;
    }
  }
}

auto NodeIdTraversal::PerformTask(
    DeferredDefinitionWorklist::EnterNestedDeferredDefinitionScope&& enter)
    -> void {
  CARBON_CHECK(enter.suspended_name,
               "Entering a scope with no suspension information.");
  context_->decl_name_stack().Restore(std::move(*enter.suspended_name));
}

auto NodeIdTraversal::PerformTask(
    DeferredDefinitionWorklist::LeaveNestedDeferredDefinitionScope&& /*leave*/)
    -> void {
  context_->decl_name_stack().PopScope();
}

auto NodeIdTraversal::PerformTask(
    DeferredDefinitionWorklist::CheckSkippedDefinition&& parse_definition)
    -> void {
  auto& [definition_index, suspended_fn] = parse_definition;
  const auto& definition_info =
      context_->parse_tree().deferred_definitions().Get(definition_index);
  HandleFunctionDefinitionResume(*context_, definition_info.start_id,
                                 std::move(suspended_fn));
  auto range = Parse::Tree::PostorderIterator::MakeRange(
      definition_info.start_id, definition_info.definition_id);
  chunks_.push_back({.it = range.begin() + 1,
                     .end = range.end(),
                     .next_definition = next_deferred_definition_.index(),
                     .checking_deferred_definitions = false,
                     .first_worklist_index = worklist_.size(),
                     .next_worklist_index = worklist_.size()});
  ++definition_index.index;
  next_deferred_definition_.SkipTo(definition_index);
}

NodeIdTraversal::NextDeferredDefinitionCache::NextDeferredDefinitionCache(
    const Parse::Tree* tree)
    : tree_(tree) {
  SkipTo(Parse::DeferredDefinitionIndex(0));
}

// Set the specified deferred definition index as being the next one that
// will be encountered.
auto NodeIdTraversal::NextDeferredDefinitionCache::SkipTo(
    Parse::DeferredDefinitionIndex next_index) -> void {
  index_ = next_index;
  if (static_cast<size_t>(index_.index) ==
      tree_->deferred_definitions().size()) {
    start_id_ = Parse::NodeId::None;
  } else {
    start_id_ = tree_->deferred_definitions().Get(index_).start_id;
  }
}

}  // namespace Carbon::Check

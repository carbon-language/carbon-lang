// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NODE_ID_TRAVERSAL_H_
#define CARBON_TOOLCHAIN_CHECK_NODE_ID_TRAVERSAL_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/check/context.h"
#include "toolchain/check/deferred_definition_worklist.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Check {

// A traversal of the node IDs in the parse tree, in the order in which we need
// to check them.
class NodeIdTraversal {
 public:
  // `context` must not be null.
  explicit NodeIdTraversal(Context* context, llvm::raw_ostream* vlog_stream);

  // Finds the next `NodeId` to type-check. Returns nullopt if the traversal is
  // complete.
  auto Next() -> std::optional<Parse::NodeId>;

  // Performs any processing necessary after we type-check a node.
  auto Handle(Parse::NodeKind parse_kind) -> void;

 private:
  // State used to track the next deferred function definition that we will
  // encounter and need to reorder.
  class NextDeferredDefinitionCache {
   public:
    explicit NextDeferredDefinitionCache(const Parse::Tree* tree);

    // Set the specified deferred definition index as being the next one that
    // will be encountered.
    auto SkipTo(Parse::DeferredDefinitionIndex next_index) -> void;

    // Returns the index of the next deferred definition to be encountered.
    auto index() const -> Parse::DeferredDefinitionIndex { return index_; }

    // Returns the ID of the start node of the next deferred definition.
    auto start_id() const -> Parse::NodeId { return start_id_; }

   private:
    const Parse::Tree* tree_;
    Parse::DeferredDefinitionIndex index_ =
        Parse::DeferredDefinitionIndex::None;
    Parse::NodeId start_id_ = Parse::NodeId::None;
  };

  // A chunk of the parse tree that we need to type-check.
  struct Chunk {
    Parse::Tree::PostorderIterator it;
    Parse::Tree::PostorderIterator end;
    // The next definition that will be encountered after this chunk completes.
    Parse::DeferredDefinitionIndex next_definition;
    // Whether we are currently checking deferred definitions, rather than the
    // tokens of this chunk. If so, we'll pull tasks off `worklist` and execute
    // them until we're done with this batch of deferred definitions. Otherwise,
    // we'll pull node IDs from `*it` until it reaches `end`.
    bool checking_deferred_definitions = false;
  };

  // Re-enter a nested deferred definition scope.
  auto PerformTask(
      DeferredDefinitionWorklist::EnterDeferredDefinitionScope&& enter) -> void;

  // Leave a nested or top-level deferred definition scope.
  auto PerformTask(
      DeferredDefinitionWorklist::LeaveDeferredDefinitionScope&& leave) -> void;

  // Resume checking a deferred definition.
  auto PerformTask(
      DeferredDefinitionWorklist::CheckSkippedDefinition&& parse_definition)
      -> void;

  Context* context_;
  NextDeferredDefinitionCache next_deferred_definition_;
  DeferredDefinitionWorklist worklist_;
  llvm::SmallVector<Chunk> chunks_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NODE_ID_TRAVERSAL_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/initializing_expression.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Materialize a temporary to hold the result of the given expression if it is
// an initializing expression.
auto MaterializeIfInitializing(Context& context, SemIR::NodeId expr_id)
    -> SemIR::NodeId {
  if (GetExpressionCategory(context.semantics_ir(), expr_id) ==
      SemIR::ExpressionCategory::Initializing) {
    return FinalizeTemporary(context, expr_id, /*discarded=*/false);
  }
  return expr_id;
}

// Given an initializing expression, find its return slot. Returns `Invalid` if
// there is no return slot, because the initialization is not performed in
// place.
static auto FindReturnSlotForInitializer(SemIR::File& semantics_ir,
                                         SemIR::NodeId init_id)
    -> SemIR::NodeId {
  SemIR::Node init = semantics_ir.GetNode(init_id);
  switch (init.kind()) {
    default:
      CARBON_FATAL() << "Initialization from unexpected node " << init;

    case SemIR::NodeKind::StructInit:
    case SemIR::NodeKind::TupleInit:
      // TODO: Track a return slot for these initializers.
      CARBON_FATAL() << init
                     << " should be created with its return slot already "
                        "filled in properly";

    case SemIR::NodeKind::InitializeFrom: {
      auto [src_id, dest_id] = init.GetAsInitializeFrom();
      return dest_id;
    }

    case SemIR::NodeKind::Call: {
      auto [refs_id, callee_id] = init.GetAsCall();
      if (!semantics_ir.GetFunction(callee_id).return_slot_id.is_valid()) {
        return SemIR::NodeId::Invalid;
      }
      return semantics_ir.GetNodeBlock(refs_id).back();
    }

    case SemIR::NodeKind::ArrayInit: {
      auto [src_id, refs_id] = init.GetAsArrayInit();
      return semantics_ir.GetNodeBlock(refs_id).back();
    }
  }
}

auto MarkInitializerFor(Context& context, SemIR::NodeId init_id,
                        SemIR::NodeId target_id, PendingBlock& target_block)
    -> void {
  auto return_slot_id =
      FindReturnSlotForInitializer(context.semantics_ir(), init_id);
  if (return_slot_id.is_valid()) {
    // Replace the temporary in the return slot with a reference to our target.
    CARBON_CHECK(context.semantics_ir().GetNode(return_slot_id).kind() ==
                 SemIR::NodeKind::TemporaryStorage)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << context.semantics_ir().GetNode(return_slot_id);
    target_block.MergeReplacing(return_slot_id, target_id);
  }
}

auto FinalizeTemporary(Context& context, SemIR::NodeId init_id, bool discarded)
    -> SemIR::NodeId {
  auto return_slot_id =
      FindReturnSlotForInitializer(context.semantics_ir(), init_id);
  if (return_slot_id.is_valid()) {
    // The return slot should already have a materialized temporary in it.
    CARBON_CHECK(context.semantics_ir().GetNode(return_slot_id).kind() ==
                 SemIR::NodeKind::TemporaryStorage)
        << "Return slot for initializer does not contain a temporary; "
        << "initialized multiple times? Have "
        << context.semantics_ir().GetNode(return_slot_id);
    auto init = context.semantics_ir().GetNode(init_id);
    return context.AddNode(SemIR::Node::Temporary::Make(
        init.parse_node(), init.type_id(), return_slot_id, init_id));
  }

  if (discarded) {
    // Don't invent a temporary that we're going to discard.
    return SemIR::NodeId::Invalid;
  }

  // The initializer has no return slot, but we want to produce a temporary
  // object. Materialize one now.
  // TODO: Consider using an invalid ID to mean that we immediately
  // materialize and initialize a temporary, rather than two separate
  // nodes.
  auto init = context.semantics_ir().GetNode(init_id);
  auto temporary_id = context.AddNode(
      SemIR::Node::TemporaryStorage::Make(init.parse_node(), init.type_id()));
  return context.AddNode(SemIR::Node::Temporary::Make(
      init.parse_node(), init.type_id(), temporary_id, init_id));
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/control_flow.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::NodeId node_id, Args... args)
    -> SemIR::InstBlockId {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return SemIR::InstBlockId::Unreachable;
  }
  auto block_id = context.inst_blocks().AddPlaceholder();
  AddInst<BranchNode>(context, node_id, {block_id, args...});
  return block_id;
}

auto AddDominatedBlockAndBranch(Context& context, Parse::NodeId node_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(context, node_id);
}

auto AddDominatedBlockAndBranchWithArg(Context& context, Parse::NodeId node_id,
                                       SemIR::InstId arg_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(context, node_id,
                                                              arg_id);
}

auto AddDominatedBlockAndBranchIf(Context& context, Parse::NodeId node_id,
                                  SemIR::InstId cond_id) -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(context, node_id,
                                                         cond_id);
}

auto AddConvergenceBlockAndPush(Context& context, Parse::NodeId node_id,
                                int num_blocks) -> void {
  CARBON_CHECK(num_blocks >= 2, "no convergence");

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (context.inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = context.inst_blocks().AddPlaceholder();
      }
      CARBON_CHECK(node_id.has_value());
      AddInst<SemIR::Branch>(context, node_id, {.target_id = new_block_id});
    }
    context.inst_block_stack().Pop();
  }
  context.inst_block_stack().Push(new_block_id);
  context.region_stack().AddToRegion(new_block_id, node_id);
}

auto AddConvergenceBlockWithArgAndPush(
    Context& context, Parse::NodeId node_id,
    std::initializer_list<SemIR::InstId> block_args) -> SemIR::InstId {
  CARBON_CHECK(block_args.size() >= 2, "no convergence");

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (context.inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = context.inst_blocks().AddPlaceholder();
      }
      AddInst<SemIR::BranchWithArg>(
          context, node_id, {.target_id = new_block_id, .arg_id = arg_id});
    }
    context.inst_block_stack().Pop();
  }
  context.inst_block_stack().Push(new_block_id);
  context.region_stack().AddToRegion(new_block_id, node_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id =
      context.insts().Get(*block_args.begin()).type_id();
  return AddInst<SemIR::BlockArg>(
      context, node_id, {.type_id = result_type_id, .block_id = new_block_id});
}

auto SetBlockArgResultBeforeConstantUse(Context& context,
                                        SemIR::InstId select_id,
                                        SemIR::InstId cond_id,
                                        SemIR::InstId if_true,
                                        SemIR::InstId if_false) -> void {
  CARBON_CHECK(context.insts().Is<SemIR::BlockArg>(select_id));

  // Determine the constant result based on the condition value.
  SemIR::ConstantId const_id = SemIR::ConstantId::NotConstant;
  auto cond_const_id = context.constant_values().Get(cond_id);
  if (!cond_const_id.is_concrete()) {
    // Symbolic or non-constant condition means a non-constant result.
  } else if (auto literal = context.insts().TryGetAs<SemIR::BoolLiteral>(
                 context.constant_values().GetInstId(cond_const_id))) {
    const_id = context.constant_values().Get(
        literal.value().value.ToBool() ? if_true : if_false);
  } else {
    CARBON_CHECK(cond_const_id == SemIR::ErrorInst::SingletonConstantId,
                 "Unexpected constant branch condition.");
    const_id = SemIR::ErrorInst::SingletonConstantId;
  }

  if (const_id.is_constant()) {
    CARBON_VLOG_TO(context.vlog_stream(), "Constant: {0} -> {1}\n",
                   context.insts().Get(select_id),
                   context.constant_values().GetInstId(const_id));
    context.constant_values().Set(select_id, const_id);
  }
}

auto IsCurrentPositionReachable(Context& context) -> bool {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = context.inst_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_inst = context.insts().Get(block_contents.back());
  return last_inst.kind().terminator_kind() !=
         SemIR::TerminatorKind::Terminator;
}

auto MaybeAddCleanupForInst(Context& context, SemIR::TypeId type_id,
                            SemIR::InstId inst_id) -> void {
  if (!context.scope_stack().PeekIsLexicalScope()) {
    // Cleanup can only occur in lexical scopes.
    return;
  }

  // TODO: Add destruction of members of ArrayType, StructType, and
  // TupleType. Includes refactoring MaybeAddCleanupForInst to remain
  // non-recursive.

  auto type_inst = context.types().GetAsInst(type_id);
  CARBON_KIND_SWITCH(type_inst) {
    case SemIR::ClassType::Kind: {
      // TODO: Figure out what destruction of classes with destroyable members
      // should look like (maybe an implicit `fn destroy` added by
      // `handle_class.cpp`?).
      auto destroy_id =
          PerformMemberAccess(context, SemIR::LocId::None, inst_id,
                              SemIR::NameId::Destroy, /*required=*/false);
      if (destroy_id.has_value()) {
        context.scope_stack().destroy_id_stack().AppendToTop(destroy_id);
      }
      break;
    }

    default:
      // Not interesting storage for destruction.
      return;
  }
}

// Common support for cleanup blocks.
static auto AddCleanupBlock(Context& context) -> void {
  auto destroy_ids = context.scope_stack().destroy_id_stack().PeekArray();

  // If there's nothing to destroy, add the final instruction to the current
  // block.
  if (destroy_ids.empty()) {
    return;
  }

  for (auto destroy_id : llvm::reverse(destroy_ids)) {
    PerformCall(context, SemIR::LocId::None, destroy_id, {});
  }
}

auto AddReturnCleanupBlock(
    Context& context,
    typename decltype(SemIR::Return::Kind)::TypedNodeId node_id) -> void {
  AddCleanupBlock(context);
  AddInst(context, node_id, SemIR::Return{});
}

}  // namespace Carbon::Check

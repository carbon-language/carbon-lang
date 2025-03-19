// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INST_H_
#define CARBON_TOOLCHAIN_CHECK_INST_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Adds an instruction to the current block, returning the produced ID.
auto AddInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for AddInst with typed nodes.
template <typename InstT, typename LocT>
auto AddInst(Context& context, LocT loc, InstT inst) -> SemIR::InstId {
  return AddInst(context, SemIR::LocIdAndInst(loc, inst));
}

// Pushes a parse tree node onto the stack, storing the SemIR::Inst as the
// result.
template <typename InstT>
  requires(SemIR::Internal::HasNodeId<InstT>)
auto AddInstAndPush(Context& context,
                    typename decltype(InstT::Kind)::TypedNodeId node_id,
                    InstT inst) -> void {
  context.node_stack().Push(node_id, AddInst(context, node_id, inst));
}

// Adds an instruction in no block, returning the produced ID. Should be used
// rarely.
auto AddInstInNoBlock(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for AddInstInNoBlock with typed nodes.
template <typename InstT, typename LocT>
auto AddInstInNoBlock(Context& context, LocT loc, InstT inst) -> SemIR::InstId {
  return AddInstInNoBlock(context, SemIR::LocIdAndInst(loc, inst));
}

// If the instruction has an implicit location and a constant value, returns
// the constant value's instruction ID. Otherwise, same as AddInst.
auto GetOrAddInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for GetOrAddInst with typed nodes.
template <typename InstT, typename LocT>
auto GetOrAddInst(Context& context, LocT loc, InstT inst) -> SemIR::InstId {
  return GetOrAddInst(context, SemIR::LocIdAndInst(loc, inst));
}

// Adds an instruction and enqueues it to be added to the eval block of the
// enclosing generic, returning the produced ID. The instruction is expected to
// be a dependent template instantiation action.
auto AddDependentActionInst(Context& context,
                            SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience wrapper for AddDependentActionInst.
template <typename InstT, typename LocT>
auto AddDependentActionInst(Context& context, LocT loc, InstT inst)
    -> decltype(AddDependentActionInst(context,
                                       SemIR::LocIdAndInst(loc, inst))) {
  return AddDependentActionInst(context, SemIR::LocIdAndInst(loc, inst));
}

// Adds an instruction to the current pattern block, returning the produced
// ID.
// TODO: Is it possible to remove this and pattern_block_stack, now that
// we have BeginSubpattern etc. instead?
auto AddPatternInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for AddPatternInst with typed nodes.
template <typename InstT>
  requires(SemIR::Internal::HasNodeId<InstT>)
auto AddPatternInst(Context& context,
                    typename decltype(InstT::Kind)::TypedNodeId node_id,
                    InstT inst) -> SemIR::InstId {
  return AddPatternInst(context, SemIR::LocIdAndInst(node_id, inst));
}

// Adds an instruction to the current block, returning the produced ID. The
// instruction is a placeholder that is expected to be replaced by
// `ReplaceInstBeforeConstantUse`.
auto AddPlaceholderInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for AddPlaceholderInst with typed nodes.
template <typename InstT, typename LocT>
auto AddPlaceholderInst(Context& context, LocT loc, InstT inst)
    -> SemIR::InstId {
  return AddPlaceholderInst(context, SemIR::LocIdAndInst(loc, inst));
}

// Adds an instruction in no block, returning the produced ID. Should be used
// rarely. The instruction is a placeholder that is expected to be replaced by
// `ReplaceInstBeforeConstantUse`.
auto AddPlaceholderInstInNoBlock(Context& context,
                                 SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId;

// Convenience for AddPlaceholderInstInNoBlock with typed nodes.
template <typename InstT, typename LocT>
auto AddPlaceholderInstInNoBlock(Context& context, LocT loc, InstT inst)
    -> SemIR::InstId {
  return AddPlaceholderInstInNoBlock(context, SemIR::LocIdAndInst(loc, inst));
}

// Replaces the instruction at `inst_id` with `loc_id_and_inst`. The
// instruction is required to not have been used in any constant evaluation,
// either because it's newly created and entirely unused, or because it's only
// used in a position that constant evaluation ignores, such as a return slot.
auto ReplaceLocIdAndInstBeforeConstantUse(Context& context,
                                          SemIR::InstId inst_id,
                                          SemIR::LocIdAndInst loc_id_and_inst)
    -> void;

// Replaces the instruction at `inst_id` with `inst`, not affecting location.
// The instruction is required to not have been used in any constant
// evaluation, either because it's newly created and entirely unused, or
// because it's only used in a position that constant evaluation ignores, such
// as a return slot.
auto ReplaceInstBeforeConstantUse(Context& context, SemIR::InstId inst_id,
                                  SemIR::Inst inst) -> void;

// Replaces the instruction at `inst_id` with `inst`, not affecting location.
// The instruction is required to not change its constant value.
auto ReplaceInstPreservingConstantValue(Context& context, SemIR::InstId inst_id,
                                        SemIR::Inst inst) -> void;

// Sets only the parse node of an instruction. This is only used when setting
// the parse node of an imported namespace. Versus
// ReplaceInstBeforeConstantUse, it is safe to use after the namespace is used
// in constant evaluation. It's exposed this way mainly so that `insts()` can
// remain const.
auto SetNamespaceNodeId(Context& context, SemIR::InstId inst_id,
                        Parse::NodeId node_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INST_H_

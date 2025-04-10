// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/inst.h"

#include "common/vlog.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::Check {

// Finish producing an instruction. Set its constant value, and register it in
// any applicable instruction lists.
static auto FinishInst(Context& context, SemIR::InstId inst_id,
                       SemIR::Inst inst) -> void {
  GenericRegionStack::DependencyKind dep_kind =
      GenericRegionStack::DependencyKind::None;

  // If the instruction has a symbolic constant type, track that we need to
  // substitute into it.
  if (context.constant_values().DependsOnGenericParameter(
          context.types().GetConstantId(inst.type_id()))) {
    dep_kind |= GenericRegionStack::DependencyKind::SymbolicType;
  }

  // If the instruction has a constant value, compute it.
  auto const_id = TryEvalInstUnsafe(context, inst_id, inst);
  context.constant_values().Set(inst_id, const_id);
  if (const_id.is_constant()) {
    CARBON_VLOG_TO(context.vlog_stream(), "Constant: {0} -> {1}\n", inst,
                   context.constant_values().GetInstId(const_id));

    // If the constant value is symbolic, track that we need to substitute into
    // it.
    if (context.constant_values().DependsOnGenericParameter(const_id)) {
      dep_kind |= GenericRegionStack::DependencyKind::SymbolicConstant;
    }
  }

  // Template-dependent instructions are handled separately by
  // `AddDependentActionInst`.
  CARBON_CHECK(
      inst.kind().constant_kind() != SemIR::InstConstantKind::InstAction,
      "Use AddDependentActionInst to add an action instruction");

  // Keep track of dependent instructions.
  if (dep_kind != GenericRegionStack::DependencyKind::None) {
    context.generic_region_stack().AddDependentInst(
        {.inst_id = inst_id, .kind = dep_kind});
  }
}

auto AddInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(context, loc_id_and_inst);
  context.inst_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto AddInstInNoBlock(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = context.sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG_TO(context.vlog_stream(), "AddInst: {0}\n", loc_id_and_inst.inst);
  FinishInst(context, inst_id, loc_id_and_inst.inst);
  return inst_id;
}

auto AddDependentActionInst(Context& context,
                            SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = context.sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG_TO(context.vlog_stream(), "AddDependentActionInst: {0}\n",
                 loc_id_and_inst.inst);

  // Set the constant value of this instruction to point back to itself.
  auto const_id = context.constant_values().AddSymbolicConstant(
      {.inst_id = inst_id,
       .generic_id = SemIR::GenericId::None,
       .index = SemIR::GenericInstIndex::None,
       .dependence = SemIR::ConstantDependence::Template});
  context.constant_values().Set(inst_id, const_id);

  // Register the instruction to be added to the eval block.
  context.generic_region_stack().AddDependentInst(
      {.inst_id = inst_id,
       .kind = GenericRegionStack::DependencyKind::Template});
  return inst_id;
}

auto AddPatternInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(context, loc_id_and_inst);
  context.pattern_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto GetOrAddInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  CARBON_CHECK(!loc_id_and_inst.inst.kind().has_cleanup());

  auto handle_constant_id = [&](SemIR::ConstantId const_id) -> SemIR::InstId {
    CARBON_CHECK(const_id.has_value());

    // If we didn't produce a constant value for the instruction, we have to add
    // the instruction.
    if (!const_id.is_constant()) {
      return SemIR::InstId::None;
    }

    CARBON_VLOG_TO(context.vlog_stream(), "GetOrAddInst: constant: {0}\n",
                   loc_id_and_inst.inst);
    return context.constant_values().GetInstId(const_id);
  };

  // If the instruction is implicit, produce its constant value instead if
  // possible.
  if (loc_id_and_inst.loc_id.is_implicit()) {
    switch (loc_id_and_inst.inst.kind().constant_needs_inst_id()) {
      case SemIR::InstConstantNeedsInstIdKind::No: {
        // Evaluation doesn't need an InstId. Just do it.
        auto const_id = TryEvalInstUnsafe(context, SemIR::InstId::None,
                                          loc_id_and_inst.inst);
        if (auto result_inst_id = handle_constant_id(const_id);
            result_inst_id.has_value()) {
          return result_inst_id;
        }
        break;
      }

      case SemIR::InstConstantNeedsInstIdKind::DuringEvaluation: {
        // Evaluation temporarily needs an InstId. Add one for now.
        auto inst_id = AddInstInNoBlock(context, loc_id_and_inst);
        auto const_id = context.constant_values().Get(inst_id);
        if (auto result_inst_id = handle_constant_id(const_id);
            result_inst_id.has_value()) {
          // TODO: We didn't end up needing the `inst_id` instruction. Consider
          // removing it from `insts` if it's still the most recently added
          // instruction.
          CARBON_CHECK(result_inst_id != inst_id);
          return result_inst_id;
        }
        context.inst_block_stack().AddInstId(inst_id);
        return inst_id;
      }

      case SemIR::InstConstantNeedsInstIdKind::Permanent: {
        // Evaluation needs a permanent InstId. Add the instruction.
        break;
      }
    }
  }

  // TODO: For an implicit instruction, this reattempts evaluation.
  return AddInst(context, loc_id_and_inst);
}

auto EvalOrAddInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::ConstantId {
  CARBON_CHECK(!loc_id_and_inst.inst.kind().has_cleanup());

  switch (loc_id_and_inst.inst.kind().constant_needs_inst_id()) {
    case SemIR::InstConstantNeedsInstIdKind::No: {
      // Evaluation doesn't need an InstId. Just do it.
      return TryEvalInstUnsafe(context, SemIR::InstId::None,
                               loc_id_and_inst.inst);
    }

    case SemIR::InstConstantNeedsInstIdKind::DuringEvaluation: {
      // Evaluation temporarily needs an InstId. Add one for now.
      auto inst_id = AddInstInNoBlock(context, loc_id_and_inst);
      // TODO: Consider removing `inst_id` from `insts` if it's still the most
      // recently added instruction.
      return context.constant_values().Get(inst_id);
    }

    case SemIR::InstConstantNeedsInstIdKind::Permanent: {
      // Evaluation needs a permanent InstId. Add the instruction.
      auto inst_id = AddInst(context, loc_id_and_inst);
      return context.constant_values().Get(inst_id);
    }
  }
}

auto AddPlaceholderInstInNoBlock(Context& context,
                                 SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = context.sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG_TO(context.vlog_stream(), "AddPlaceholderInst: {0}\n",
                 loc_id_and_inst.inst);
  context.constant_values().Set(inst_id, SemIR::ConstantId::None);
  return inst_id;
}

auto AddPlaceholderInst(Context& context, SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInstInNoBlock(context, loc_id_and_inst);
  context.inst_block_stack().AddInstId(inst_id);
  return inst_id;
}

auto ReplaceLocIdAndInstBeforeConstantUse(Context& context,
                                          SemIR::InstId inst_id,
                                          SemIR::LocIdAndInst loc_id_and_inst)
    -> void {
  context.sem_ir().insts().SetLocIdAndInst(inst_id, loc_id_and_inst);
  CARBON_VLOG_TO(context.vlog_stream(), "ReplaceInst: {0} -> {1}\n", inst_id,
                 loc_id_and_inst.inst);
  FinishInst(context, inst_id, loc_id_and_inst.inst);
}

auto ReplaceInstBeforeConstantUse(Context& context, SemIR::InstId inst_id,
                                  SemIR::Inst inst) -> void {
  context.sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG_TO(context.vlog_stream(), "ReplaceInst: {0} -> {1}\n", inst_id,
                 inst);
  FinishInst(context, inst_id, inst);
}

auto ReplaceInstPreservingConstantValue(Context& context, SemIR::InstId inst_id,
                                        SemIR::Inst inst) -> void {
  auto old_const_id = context.constant_values().Get(inst_id);
  context.sem_ir().insts().Set(inst_id, inst);
  CARBON_VLOG_TO(context.vlog_stream(), "ReplaceInst: {0} -> {1}\n", inst_id,
                 inst);
  auto new_const_id = TryEvalInstUnsafe(context, inst_id, inst);
  CARBON_CHECK(old_const_id == new_const_id);
}

auto SetNamespaceNodeId(Context& context, SemIR::InstId inst_id,
                        Parse::NodeId node_id) -> void {
  context.sem_ir().insts().SetLocId(inst_id, SemIR::LocId(node_id));
}

}  // namespace Carbon::Check

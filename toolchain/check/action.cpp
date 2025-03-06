// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/action.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

static auto OperandIsDependent(Context& context, SemIR::ConstantId const_id)
    -> bool {
  // A type operand makes the instruction dependent if it is a
  // template-dependent constant.
  if (!const_id.is_symbolic()) {
    return false;
  }
  return context.constant_values().GetSymbolicConstant(const_id).dependence ==
         SemIR::ConstantDependence::Template;
}

auto OperandIsDependent(Context& context, SemIR::TypeId type_id) -> bool {
  // A type operand makes the instruction dependent if it is a
  // template-dependent type.
  return OperandIsDependent(context, context.types().GetConstantId(type_id));
}

auto OperandIsDependent(Context& context, SemIR::MetaInstId inst_id)
    -> bool {
  // An instruction operand makes the instruction dependent if its type or
  // constant value is dependent.
  return OperandIsDependent(context, context.insts().Get(inst_id).type_id()) ||
         OperandIsDependent(context, context.constant_values().Get(inst_id));
}

static auto OperandIsDependent(Context& context, SemIR::IdKind kind,
                               int32_t arg) -> bool {
  if (kind == SemIR::IdKind::For<SemIR::MetaInstId>) {
    return OperandIsDependent(context, SemIR::MetaInstId(arg));
  }
  if (kind == SemIR::IdKind::For<SemIR::TypeId>) {
    return OperandIsDependent(context, SemIR::TypeId(arg));
  }
  if (kind == SemIR::IdKind::For<SemIR::AbsoluteInstId> ||
      kind == SemIR::IdKind::For<SemIR::NameId>) {
    return false;
  }
  // TODO: Properly handle different argument kinds.
  CARBON_FATAL("Unexpected argument kind for action");
}

auto ActionIsDependent(Context& context, SemIR::Inst action_inst) -> bool {
  if (OperandIsDependent(context, action_inst.type_id())) {
    return true;
  }
  auto [arg0_kind, arg1_kind] = action_inst.ArgKinds();
  return OperandIsDependent(context, arg0_kind, action_inst.arg0()) ||
         OperandIsDependent(context, arg1_kind, action_inst.arg1());
}

auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeId result_type_id) -> SemIR::InstId {
  auto inst_id = AddDependentActionInst(context, action);
  if (!result_type_id.has_value()) {
    auto type_inst_id = AddDependentActionInst(
        context,
        SemIR::LocIdAndInst(
            action.loc_id,
            SemIR::TypeOfInst{.type_id = SemIR::TypeType::SingletonTypeId,
                              .inst_id = inst_id}));
    result_type_id = context.types().GetTypeIdForTypeInstId(type_inst_id);
  }
  return AddInst(
      context, action.loc_id,
      SemIR::SpliceInst{.type_id = result_type_id, .inst_id = inst_id});
}

auto BeginPerformDelayedAction(Context& context) -> void {
  // Push an `InstBlock` to hold any instructions created by the action.
  // Note that we assume that actions don't need to create multiple blocks. If
  // this changes, we should push a region too.
  context.inst_block_stack().Push();
}

auto EndPerformDelayedAction(Context& context, SemIR::InstId result_id) -> SemIR::InstId {
  // If the only created instruction is the result, then we can use it directly.
  auto contents = context.inst_block_stack().PeekCurrentBlockContents();
  if (contents.size() == 1 && contents[0] == result_id) {
    context.inst_block_stack().PopAndDiscard();
    return result_id;
  }

  // Otherwise, create a splice_block to represent the sequence of instructions
  // created by the action.
  auto result = context.insts().GetWithLocId(result_id);
  return AddInstInNoBlock(
      context, result.loc_id,
      SemIR::SpliceBlock{.type_id = result.inst.type_id(),
                         .block_id = context.inst_block_stack().Pop(),
                         .result_id = result_id});
}

}  // namespace Carbon::Check

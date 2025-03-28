// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/action.h"

#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::RefineTypeAction action) -> SemIR::InstId {
  return AddInst<SemIR::AsCompatible>(
      context, loc_id,
      {.type_id = action.inst_type_id, .source_id = action.inst_id});
}

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

auto OperandIsDependent(Context& context, SemIR::MetaInstId inst_id) -> bool {
  // An instruction operand makes the instruction dependent if its type or
  // constant value is dependent.
  return OperandIsDependent(context, context.insts().Get(inst_id).type_id()) ||
         OperandIsDependent(context, context.constant_values().Get(inst_id));
}

static auto OperandIsDependent(Context& context, SemIR::Inst::ArgAndKind arg)
    -> bool {
  switch (arg.kind) {
    case SemIR::IdKind::For<SemIR::MetaInstId>:
      return OperandIsDependent(context, arg.As<SemIR::MetaInstId>());

    case SemIR::IdKind::For<SemIR::TypeId>:
      return OperandIsDependent(context, arg.As<SemIR::TypeId>());

    case SemIR::IdKind::None:
    case SemIR::IdKind::For<SemIR::AbsoluteInstId>:
    case SemIR::IdKind::For<SemIR::NameId>:
      return false;

    default:
      // TODO: Properly handle different argument kinds.
      CARBON_FATAL("Unexpected argument kind for action");
  }
}

auto ActionIsDependent(Context& context, SemIR::Inst action_inst) -> bool {
  if (auto refine_action = action_inst.TryAs<SemIR::RefineTypeAction>()) {
    // `RefineTypeAction` can be performed whenever the type is non-dependent,
    // even if we don't know the instruction yet.
    return OperandIsDependent(context, refine_action->inst_type_id);
  }

  if (OperandIsDependent(context, action_inst.type_id())) {
    return true;
  }
  return OperandIsDependent(context, action_inst.arg0_and_kind()) ||
         OperandIsDependent(context, action_inst.arg1_and_kind());
}

static auto AddDependentActionSpliceImpl(Context& context,
                                         SemIR::LocIdAndInst action,
                                         SemIR::TypeId result_type_id)
    -> SemIR::InstId {
  auto inst_id = AddDependentActionInst(context, action);
  if (!result_type_id.has_value()) {
    auto type_inst_id = AddDependentActionInst(
        context, action.loc_id,
        SemIR::TypeOfInst{.type_id = SemIR::TypeType::SingletonTypeId,
                          .inst_id = inst_id});
    result_type_id = context.types().GetTypeIdForTypeInstId(type_inst_id);
  }
  return AddInst(
      context, action.loc_id,
      SemIR::SpliceInst{.type_id = result_type_id, .inst_id = inst_id});
}

// Refine one operand of an action. Given an argument from a template, this
// produces an argument that has the template-dependent parts replaced with
// their concrete values, so that the action doesn't need to know which specific
// it is operating on.
static auto RefineOperand(Context& context, SemIR::LocId loc_id,
                          SemIR::Inst::ArgAndKind arg) -> int32_t {
  if (auto inst_id = arg.TryAs<SemIR::MetaInstId>()) {
    auto inst = context.insts().Get(*inst_id);
    if (inst.Is<SemIR::SpliceInst>()) {
      // The argument will evaluate to the spliced instruction, which is already
      // refined.
      return arg.value;
    }

    // If the type of the action argument is dependent, refine to an instruction
    // with a concrete type.
    if (OperandIsDependent(context, inst.type_id())) {
      inst_id = AddDependentActionSpliceImpl(
          context,
          SemIR::LocIdAndInst(loc_id,
                              SemIR::RefineTypeAction{
                                  .type_id = SemIR::InstType::SingletonTypeId,
                                  .inst_id = *inst_id,
                                  .inst_type_id = inst.type_id()}),
          inst.type_id());
    }

    // TODO: Handle the case where the constant value of the instruction is
    // template-dependent.

    return inst_id->index;
  }

  return arg.value;
}

// Refine the operands of an action, ensuring that they will refer to concrete
// instructions that don't have template-dependent types.
static auto RefineOperands(Context& context, SemIR::LocId loc_id,
                           SemIR::Inst action) -> SemIR::Inst {
  auto arg0 = RefineOperand(context, loc_id, action.arg0_and_kind());
  auto arg1 = RefineOperand(context, loc_id, action.arg1_and_kind());
  action.SetArgs(arg0, arg1);
  return action;
}

auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeId result_type_id) -> SemIR::InstId {
  action.inst = RefineOperands(context, action.loc_id, action.inst);
  return AddDependentActionSpliceImpl(context, action, result_type_id);
}

auto Internal::BeginPerformDelayedAction(Context& context) -> void {
  // Push an `InstBlock` to hold any instructions created by the action.
  // Note that we assume that actions don't need to create multiple blocks. If
  // this changes, we should push a region too.
  context.inst_block_stack().Push();
}

auto Internal::EndPerformDelayedAction(Context& context,
                                       SemIR::InstId result_id)
    -> SemIR::InstId {
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

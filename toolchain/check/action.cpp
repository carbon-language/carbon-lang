// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/action.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/copy_on_write_block.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::RefineTypeAction action) -> SemIR::InstId {
  return AddInst<SemIR::AsCompatible>(
      context, loc_id,
      {.type_id =
           context.types().GetTypeIdForTypeInstId(action.inst_type_inst_id),
       .source_id = action.inst_id});
}

auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::RefineFormAction action) -> SemIR::InstId {
  auto expr_const_id = context.constant_values().Get(action.form_id);
  if (!expr_const_id.is_constant()) {
    // This is an error which will be diagnosed elsewhere, so we should just
    // get out of its way.
    return action.form_id;
  }
  auto expr =
      context.insts().Get(context.constant_values().GetInstId(expr_const_id));
  CARBON_KIND_SWITCH(expr) {
    case SemIR::InitForm::Kind:
    case SemIR::RefForm::Kind:
    case SemIR::ValueForm::Kind:
      // Primitive forms have no form operands, so the action is a no-op.
      return action.form_id;
    case CARBON_KIND(SemIR::TupleValue tuple_value): {
      SemIR::CopyOnWriteInstBlock new_inst_block(&context.sem_ir(),
                                                 tuple_value.elements_id);
      for (auto [i, element_id] : llvm::enumerate(
               context.inst_blocks().Get(tuple_value.elements_id))) {
        new_inst_block.Set(i, HandleAction<SemIR::RefineFormAction>(
                                  context, loc_id, SemIR::FormType::TypeInstId,
                                  {.type_id = SemIR::InstType::TypeId,
                                   .form_id = element_id}));
      }
      auto new_inst_block_id = new_inst_block.GetCanonical();
      if (new_inst_block_id == tuple_value.elements_id) {
        return action.form_id;
      }
      return AddInst<SemIR::TupleValue>(context, loc_id,
                                        {.type_id = SemIR::FormType::TypeId,
                                         .elements_id = new_inst_block_id});
    }
    default:
      CARBON_FATAL("Unexpected kind for non-dependent form constant {0}", expr);
  }
}

static auto OperandDependence(Context& context, SemIR::ConstantId const_id)
    -> SemIR::ConstantDependence {
  // A type operand makes the instruction dependent if it is a
  // template-dependent constant.
  if (!const_id.is_symbolic()) {
    return SemIR::ConstantDependence::None;
  }
  return context.constant_values().GetSymbolicConstant(const_id).dependence;
}

auto OperandDependence(Context& context, SemIR::TypeId type_id)
    -> SemIR::ConstantDependence {
  // A type operand makes the instruction dependent if it is a
  // template-dependent type.
  return OperandDependence(context, context.types().GetConstantId(type_id));
}

auto OperandDependence(Context& context, SemIR::InstId inst_id)
    -> SemIR::ConstantDependence {
  // An instruction operand makes the instruction dependent if its type or
  // constant value is dependent.
  return std::max(
      OperandDependence(context, context.insts().Get(inst_id).type_id()),
      OperandDependence(context, context.constant_values().Get(inst_id)));
}

static auto OperandDependence(Context& context, SemIR::MetaInstId inst_id)
    -> SemIR::ConstantDependence {
  // A meta-instruction operand makes the instruction dependent if its type or
  // constant value is dependent.
  return OperandDependence(context, SemIR::InstId{inst_id});
}

auto OperandDependence(Context& context, SemIR::TypeInstId inst_id)
    -> SemIR::ConstantDependence {
  // An instruction operand makes the instruction dependent if its type or
  // constant value is dependent. TypeInstId has type `TypeType` which is
  // concrete, so we only need to look at the constant value.
  return OperandDependence(context, context.constant_values().Get(inst_id));
}

template <typename IdT>
  requires SemIR::Internal::IsIdKindType<IdT> &&
           SameAsOneOf<IdT, SemIR::IdAndKind::NoneType, SemIR::AbsoluteInstId,
                       SemIR::CallParamIndex, SemIR::NameId>
static auto OperandDependence(Context& /*context*/, IdT /*id*/)
    -> SemIR::ConstantDependence {
  return SemIR::ConstantDependence::None;
}

template <typename BundleT>
static auto OperandDependence(Context& context,
                              SemIR::BundleId<BundleT> bundle_id)
    -> SemIR::ConstantDependence {
  return std::apply(
      [&](auto... ids) {
        return std::max({OperandDependence(context, ids)...});
      },
      context.bundles().GetAsTuple(bundle_id));
}

template <typename IdT>
  requires SemIR::Internal::IsIdKindType<IdT>
static auto OperandDependence(Context& /*context*/, IdT /*id*/)
    -> SemIR::ConstantDependence {
  // TODO: Properly handle different argument kinds.
  CARBON_FATAL("Unexpected argument kind for action: {}", IdT::Label);
}

static auto OperandDependence(Context& context, SemIR::IdAndKind arg)
    -> SemIR::ConstantDependence {
  return arg.Dispatch<SemIR::ConstantDependence>(
      [&](auto id) { return OperandDependence(context, id); });
}

auto ActionIsPerformable(Context& context, SemIR::Inst action_inst) -> bool {
  if (auto refine_action = action_inst.TryAs<SemIR::RefineTypeAction>()) {
    // `RefineTypeAction` can be performed whenever the type is not template-
    // dependent, even if we don't know the instruction yet.
    return OperandDependence(context, refine_action->inst_type_inst_id) <
           SemIR::ConstantDependence::Template;
  }

  if (auto refine_action = action_inst.TryAs<SemIR::RefineFormAction>()) {
    auto form_const_id = context.constant_values().Get(refine_action->form_id);
    auto form_id = context.constant_values().GetInstIdIfValid(form_const_id);
    if (!form_id.has_value()) {
      // This is an error which will be diagnosed elsewhere, so we should just
      // get out of its way.
      return true;
    }
    // A RefineFormAction can be performed if we can identify all of the
    // subexpressions of `form_id` that are in form positions.
    CARBON_KIND_SWITCH(context.insts().Get(form_id)) {
      case SemIR::InitForm::Kind:
      case SemIR::RefForm::Kind:
      case SemIR::ValueForm::Kind:
      case SemIR::TupleValue::Kind:
        // These inst kinds are not rewritten by constant evaluation except to
        // substitute values for their operands (i.e. their constant kind is
        // WheneverPossible, Always, or AlwaysUnique), so we can identify all of
        // their form subexpressions.
        return true;
      default:
        // All other inst kinds either can't appear in a form position, or
        // may be rewritten by constant evaluation, so we can't identify
        // their form subexpressions unless they are already concrete.
        return OperandDependence(context, form_const_id) ==
               SemIR::ConstantDependence::None;
    }
  }

  // A form-parameterized action is performable if we can see at least the top
  // level of its form's structure (i.e. it is not an action or a splice).
  if (auto form_parameterized_action =
          action_inst.TryAs<SemIR::AnyFormParamAction>()) {
    auto form_const_id =
        context.constant_values().Get(form_parameterized_action->form_id);
    auto form_id = context.constant_values().GetInstIdIfValid(form_const_id);
    if (!form_id.has_value()) {
      // This is an error which will be diagnosed elsewhere, so we should just
      // get out of its way.
      return true;
    }
    return !context.insts().Is<SemIR::RefineFormAction>(form_id) &&
           !context.insts().Is<SemIR::SpliceInst>(form_id);
  }

  return OperandDependence(context, action_inst.type_id()) <
             SemIR::ConstantDependence::Template &&
         OperandDependence(context, action_inst.arg0_and_kind()) <
             SemIR::ConstantDependence::Template &&
         OperandDependence(context, action_inst.arg1_and_kind()) <
             SemIR::ConstantDependence::Template;
}

static auto AddDependentActionSpliceImpl(Context& context,
                                         SemIR::LocIdAndInst action,
                                         SemIR::TypeInstId result_type_inst_id)
    -> SemIR::InstId {
  auto inst_id = AddDependentActionInst(context, action);
  if (!result_type_inst_id.has_value()) {
    result_type_inst_id = AddDependentActionTypeInst(
        context, action.loc_id,
        SemIR::TypeOfInst{.type_id = SemIR::TypeType::TypeId,
                          .inst_id = inst_id});
  }
  return AddInst(
      context, action.loc_id,
      SemIR::SpliceInst{.type_id = context.types().GetTypeIdForTypeInstId(
                            result_type_inst_id),
                        .inst_id = inst_id});
}

// Refine one operand of an action. Given an argument from a template, this
// produces an argument that has the template-dependent parts replaced with
// their concrete values, so that the action doesn't need to know which specific
// it is operating on.
static auto RefineOperand(Context& context, SemIR::LocId loc_id,
                          SemIR::IdAndKind arg) -> int32_t {
  if (auto inst_id = arg.TryAs<SemIR::MetaInstId>()) {
    auto inst = context.insts().Get(*inst_id);
    if (inst.Is<SemIR::SpliceInst>()) {
      // The argument will evaluate to the spliced instruction, which is already
      // refined.
      return arg.value();
    }

    // If the type of the action argument is dependent, refine to an instruction
    // with a concrete type.
    if (OperandDependence(context, inst.type_id()) ==
        SemIR::ConstantDependence::Template) {
      auto type_inst_id = context.types().GetTypeInstId(inst.type_id());
      inst_id = AddDependentActionSpliceImpl(
          context,
          SemIR::LocIdAndInst(
              loc_id,
              SemIR::RefineTypeAction{.type_id = GetSingletonType(
                                          context, SemIR::InstType::TypeInstId),
                                      .inst_id = *inst_id,
                                      .inst_type_inst_id = type_inst_id}),
          type_inst_id);
    }

    // TODO: Handle the case where the constant value of the instruction is
    // template-dependent.

    return inst_id->index;
  }

  return arg.value();
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
                              SemIR::TypeInstId result_type_inst_id)
    -> SemIR::InstId {
  action.inst = RefineOperands(context, action.loc_id, action.inst);
  return AddDependentActionSpliceImpl(context, action, result_type_inst_id);
}

auto Internal::BeginPerformDelayedAction(Context& context) -> void {
  // Push an `InstBlock` to hold any instructions created by the action.
  // Note that we assume that actions don't need to create multiple blocks. If
  // this changes, we should push a region too.
  context.inst_block_stack().Push();
  context.pattern_block_stack().Push();
}

auto Internal::EndPerformDelayedAction(Context& context,
                                       SemIR::InstId result_id)
    -> SemIR::InstId {
  // If the only created instruction is the result, then we can use it directly.
  auto contents = context.inst_block_stack().PeekCurrentBlockContents();
  auto pattern_contents =
      context.pattern_block_stack().PeekCurrentBlockContents();
  if ((contents == llvm::ArrayRef(result_id) && pattern_contents.empty()) ||
      (pattern_contents == llvm::ArrayRef(result_id) && contents.empty())) {
    context.inst_block_stack().PopAndDiscard();
    context.pattern_block_stack().PopAndDiscard();
    return result_id;
  }

  // Otherwise, create a splice_block to represent the sequence of instructions
  // created by the action.
  auto block_id = SemIR::InstBlockId::None;
  if (pattern_contents.empty()) {
    block_id = context.inst_block_stack().Pop();
    context.pattern_block_stack().PopAndDiscard();
  } else {
    // TODO: pattern insts can depend on non-pattern insts, so we'll probably
    // eventually need to support actions that produce both.
    CARBON_CHECK(!contents.empty());
    block_id = context.pattern_block_stack().Pop();
    context.inst_block_stack().PopAndDiscard();
  }
  auto result = context.insts().GetWithLocId(result_id);
  return AddInstInNoBlock(context, result.loc_id,
                          SemIR::SpliceBlock{.type_id = result.inst.type_id(),
                                             .block_id = block_id,
                                             .result_id = result_id});
}

}  // namespace Carbon::Check

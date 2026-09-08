// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_ACTION_H_
#define CARBON_TOOLCHAIN_CHECK_ACTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace Internal {
// Computes the `SpecificId` parameters to use for PerformAction for InstT.
template <typename InstT>
using SpecificParamsForPerformAction =
    std::conditional_t<InstT::Kind.action_needs_specific_id(),
                       auto(SemIR::SpecificId specific_id)->void, auto()->void>;

// Computes the function type to use for PerformAction for InstT.
template <typename InstT,
          typename SpecificParams = SpecificParamsForPerformAction<InstT>>
struct FunctionTypeForPerformActionImpl {
  // By default, no PerformAction function.
  using Type = auto() -> void;
};
template <typename InstT, typename... SpecificParams>
  requires(InstT::Kind.constant_kind() == SemIR::InstConstantKind::InstAction)
struct FunctionTypeForPerformActionImpl<InstT, auto(SpecificParams...)->void> {
  using Type = auto(Context& context, SpecificParams..., SemIR::LocId loc_id,
                    InstT inst) -> SemIR::InstId;
};
template <typename InstT>
using FunctionTypeForPerformAction =
    FunctionTypeForPerformActionImpl<InstT>::Type;
}  // namespace Internal

// Explicitly delete the overload generated for non-action instructions. These
// all produce the same signature, so we only need to delete it once.
auto PerformAction() -> void = delete;

// Performs an action. Each PerformAction implementation lives with the code
// that creates and defines the action. For an instruction whose constant kind
// is InstAction, an overload should be provided with the signature:
//
//   auto PerformAction(Context& context, SemIR::LocId loc_id, InstT inst)
//       -> SemIR::InstId;
//
// or if action_needs_specific_id = true is specified when defining the
// instruction kind, the signature:
//
//   auto PerformAction(Context& context, SemIR::SpecificId specific_id,
//                      SemIR::LocId loc_id, InstT inst)
//       -> SemIR::InstId;
//
// that returns the value that should be used as the result of evaluating the
// instructions produced by the action. Any instructions generated during
// `PerformAction` will be spliced into the code at the point where the action
// was created.
#define CARBON_SEM_IR_INST_KIND(Name) \
  Internal::FunctionTypeForPerformAction<SemIR::Name> PerformAction;
#include "toolchain/sem_ir/inst_kind.def"

// Determines whether the given action can be performed immediately (i.e.
// whether it is non-template-dependent).
auto ActionIsPerformable(Context& context, SemIR::Inst action_inst,
                         SemIR::SpecificId specific_id) -> bool;

// Returns the constant-dependence of `inst_id` (i.e. the maximum of the
// constant-dependences of its type and its value).
auto OperandDependence(Context& context, SemIR::InstId inst_id)
    -> SemIR::ConstantDependence;
auto OperandDependence(Context& context, SemIR::TypeInstId inst_id)
    -> SemIR::ConstantDependence;

// Returns the constant-dependence of `type_id` (i.e. the constant-dependence
// of the corresponding type constant).
auto OperandDependence(Context& context, SemIR::TypeId type_id)
    -> SemIR::ConstantDependence;

// Adds an instruction to the current block to splice in the result of
// performing a dependent action.
auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeInstId result_type_inst_id)
    -> SemIR::InstId;

// Convenience wrapper for `AddDependentActionSplice`.
template <typename LocT, typename InstT>
auto AddDependentActionSplice(Context& context, LocT loc, InstT inst,
                              SemIR::TypeInstId result_type_inst_id)
    -> SemIR::InstId {
  return AddDependentActionSplice(context, SemIR::LocIdAndInst(loc, inst),
                                  result_type_inst_id);
}

// Handles a new action if necessary. If the action is not dependent, returns
// InstId::None. Otherwise, adds the action to the enclosing template's eval
// block and creates an instruction to splice in the result of the action.
// `result_type_inst_id` is the type of inst produced by the action. If not
// known, it can be set to `None`, and a `TypeOfInst` instruction will be added
// to act as the type of the splice.
template <typename ActionT, typename LocIdT>
auto AddActionSpliceIfDependent(Context& context, LocIdT loc_id,
                                SemIR::TypeInstId expected_result_type_inst_id,
                                ActionT action_inst) -> SemIR::InstId {
  CARBON_CHECK(action_inst.type_id == SemIR::InstType::TypeId);
  if (ActionIsPerformable(context, action_inst, SemIR::SpecificId::None)) {
    return SemIR::InstId::None;
  }
  return AddDependentActionSplice(context,
                                  SemIR::LocIdAndInst::RuntimeVerified(
                                      context.sem_ir(), loc_id, action_inst),
                                  expected_result_type_inst_id);
}

// Handles a new action. If the action is not dependent, it is performed
// immediately. Otherwise, adds the action to the enclosing template's eval
// block and creates an instruction to splice in the result of the action.
// `result_type_inst_id` is the type of inst produced by the action. If not
// known, it can be set to `None`, and a `TypeOfInst` instruction will be added
// to act as the type of the splice.
template <typename ActionT, typename LocIdT>
auto HandleAction(Context& context, LocIdT loc_id,
                  SemIR::TypeInstId expected_result_type_inst_id,
                  ActionT action_inst) -> SemIR::InstId {
  if (auto splice_inst_id = AddActionSpliceIfDependent(
          context, loc_id, expected_result_type_inst_id, action_inst);
      splice_inst_id.has_value()) {
    return splice_inst_id;
  }

  auto expected_result_type_id =
      expected_result_type_inst_id.has_value()
          ? context.types().GetTypeIdForTypeInstId(expected_result_type_inst_id)
          : SemIR::TypeId::None;
  auto result_id = PerformAction(context, loc_id, action_inst);
  auto result_type_id = context.insts().Get(result_id).type_id();
  CARBON_CHECK(expected_result_type_id == SemIR::TypeId::None ||
               result_type_id == SemIR::ErrorInst::TypeId ||
               result_type_id == expected_result_type_id);
  return result_id;
}

namespace Internal {
// Performs setup steps for performing a delayed action. This is an
// implementation detail of PerformDelayedAction and should not be called
// directly.
auto BeginPerformDelayedAction(Context& context) -> void;

// Performs cleanup steps for performing a delayed action. This is an
// implementation detail of PerformDelayedAction and should not be called
// directly.
auto EndPerformDelayedAction(Context& context, SemIR::InstId result_id)
    -> SemIR::InstId;
}  // namespace Internal

// Performs an action as a result of evaluation of a template's eval block.
template <typename ActionT>
auto PerformDelayedAction(Context& context, SemIR::SpecificId specific_id,
                          SemIR::LocId loc_id, ActionT action_inst)
    -> SemIR::InstId {
  if (!ActionIsPerformable(context, action_inst, specific_id)) {
    return SemIR::InstId::None;
  }
  Internal::BeginPerformDelayedAction(context);
  auto inst_id = SemIR::InstId::None;
  if constexpr (ActionT::Kind.action_needs_specific_id()) {
    inst_id = PerformAction(context, specific_id, loc_id, action_inst);
  } else {
    inst_id = PerformAction(context, loc_id, action_inst);
  }
  return Internal::EndPerformDelayedAction(context, inst_id);
}

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_ACTION_H_

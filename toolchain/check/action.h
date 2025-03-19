// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_ACTION_H_
#define CARBON_TOOLCHAIN_CHECK_ACTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Performs a member access action. Defined in member_access.cpp.
auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::AccessMemberAction action) -> SemIR::InstId;

// Performs a conversion action. Defined in convert.cpp.
auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::ConvertToValueAction action) -> SemIR::InstId;

// Performs a type refinement action, by creating a conversion from an
// instruction with a template-dependent symbolic type to the corresponding
// instantiated type.
auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::RefineTypeAction action) -> SemIR::InstId;

// Determines whether the given action depends on a template parameter in a way
// that means it cannot be performed immediately.
auto ActionIsDependent(Context& context, SemIR::Inst action_inst) -> bool;

// Determines whether the given action operand depends on a template parameter
// in a way that means the action cannot be performed immediately.
auto OperandIsDependent(Context& context, SemIR::MetaInstId inst_id) -> bool;

// Determines whether the given type depends on a template parameter
// in a way that means the action cannot be performed immediately.
auto OperandIsDependent(Context& context, SemIR::TypeId type_id) -> bool;

// Adds an instruction to the current block to splice in the result of
// performing a dependent action.
auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeId result_type_id) -> SemIR::InstId;

// Convenience wrapper for `AddDependentActionSplice`.
template <typename LocT, typename InstT>
auto AddDependentActionSplice(Context& context, LocT loc, InstT inst,
                              SemIR::TypeId result_type_id) -> SemIR::InstId {
  return AddDependentActionSplice(context, SemIR::LocIdAndInst(loc, inst),
                                  result_type_id);
}

// Handles a new action. If the action is not dependent, it is performed
// immediately. Otherwise, adds the action to the enclosing template's eval
// block and creates an instruction to splice in the result of the action.
template <typename ActionT>
auto HandleAction(Context& context, SemIR::LocId loc_id, ActionT action_inst,
                  SemIR::TypeId result_type_id = SemIR::TypeId::None)
    -> SemIR::InstId {
  if (ActionIsDependent(context, action_inst) ||
      OperandIsDependent(context, result_type_id)) {
    return AddDependentActionSplice(
        context, SemIR::LocIdAndInst(loc_id, action_inst), result_type_id);
  }

  return PerformAction(context, loc_id, action_inst);
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
auto PerformDelayedAction(Context& context, SemIR::LocId loc_id,
                          ActionT action_inst) -> SemIR::InstId {
  if (ActionIsDependent(context, action_inst)) {
    return SemIR::InstId::None;
  }
  Internal::BeginPerformDelayedAction(context);
  auto inst_id = PerformAction(context, loc_id, action_inst);
  return Internal::EndPerformDelayedAction(context, inst_id);
}

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_ACTION_H_

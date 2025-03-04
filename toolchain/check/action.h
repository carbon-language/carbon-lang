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

// Determines whether the given action depends on a template parameter in a way
// that means it cannot be performed immediately.
auto ActionIsDependent(Context& context, SemIR::Inst action_inst) -> bool;

// Determines whether the given action operand depends on a template parameter
// in a way that means the action cannot be performed immediately.
auto OperandIsDependent(Context& context, SemIR::MetaInstId inst_id)
    -> bool;

// Adds an instruction to the current block to splice in the result of
// performing a dependent action.
auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeId result_type_id) -> SemIR::InstId;

template <typename ActionT>
auto HandleAction(Context& context, SemIR::LocId loc_id, ActionT action_inst,
                  SemIR::TypeId result_type_id = SemIR::TypeId::None)
    -> SemIR::InstId {
  if (ActionIsDependent(context, action_inst)) {
    return AddDependentActionSplice(
        context, SemIR::LocIdAndInst(loc_id, action_inst), result_type_id);
  }

  return PerformAction(context, loc_id, action_inst);
}

template <typename ActionT>
auto PerformDelayedAction(Context& context, SemIR::LocId loc_id,
                          ActionT action_inst) -> SemIR::InstId {
  if (ActionIsDependent(context, action_inst)) {
    return SemIR::InstId::None;
  }
  // TODO: Push an inst block and form a splice_block instruction if needed.
  return PerformAction(context, loc_id, action_inst);
}

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_ACTION_H_

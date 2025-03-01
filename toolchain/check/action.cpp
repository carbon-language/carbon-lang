// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/action.h"

namespace Carbon::Check {

auto ActionIsDependent(Context& context, SemIR::Inst action_inst) -> bool {
  (void)context;
  (void)action_inst;
  return false;
}

auto AddDependentActionSplice(Context& context, SemIR::LocIdAndInst action,
                              SemIR::TypeId result_type_id) -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(context, action);
  if (!result_type_id.has_value()) {
    result_type_id = context.types().GetTypeIdForTypeInstId(AddInstInNoBlock(
        context, action.loc_id,
        SemIR::TypeOfInst{.type_id = SemIR::TypeType::SingletonTypeId,
                          .inst_id = inst_id}));
  }
  return AddInst(
      context, action.loc_id,
      SemIR::SpliceInst{.type_id = result_type_id, .inst_id = inst_id});
}

}  // namespace Carbon::Check

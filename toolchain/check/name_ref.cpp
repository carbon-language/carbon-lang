// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_ref.h"

#include "toolchain/check/inst.h"

namespace Carbon::Check {

auto BuildNameRef(Context& context, SemIR::LocId loc_id, SemIR::NameId name_id,
                  SemIR::InstId inst_id, SemIR::SpecificId specific_id)
    -> SemIR::InstId {
  auto type_id =
      SemIR::GetTypeOfInstInSpecific(context.sem_ir(), specific_id, inst_id);
  CARBON_CHECK(type_id.has_value(), "Missing type for {0}",
               context.insts().Get(inst_id));

  // If the named entity has a constant value that depends on its specific,
  // store the specific too.
  if (specific_id.has_value() &&
      context.constant_values().Get(inst_id).is_symbolic()) {
    inst_id = AddInst<SemIR::SpecificConstant>(
        context, loc_id,
        {.type_id = type_id, .inst_id = inst_id, .specific_id = specific_id});
  }

  return AddInst<SemIR::NameRef>(
      context, loc_id,
      {.type_id = type_id, .name_id = name_id, .value_id = inst_id});
}

}  // namespace Carbon::Check

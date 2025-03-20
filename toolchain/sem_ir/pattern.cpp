// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/pattern.h"

namespace Carbon::SemIR {

auto IsSelfPattern(const File& sem_ir, InstId pattern_id) -> bool {
  // Note that the public contract of GetPrettyNameFromPatternId does not
  // guarantee that this is correct; we're relying on knowledge of the
  // implementation details.
  return GetPrettyNameFromPatternId(sem_ir, pattern_id) == NameId::SelfValue;
}

auto GetPrettyNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> NameId {
  auto inst_id = pattern_id;
  auto inst = sem_ir.insts().Get(inst_id);

  if (auto var_pattern = inst.TryAs<VarPattern>()) {
    inst_id = var_pattern->subpattern_id;
    inst = sem_ir.insts().Get(inst_id);
  }

  if (auto addr_pattern = inst.TryAs<AddrPattern>()) {
    inst_id = addr_pattern->inner_id;
    inst = sem_ir.insts().Get(inst_id);
  }

  if (auto param_pattern_inst = inst.TryAs<AnyParamPattern>()) {
    inst_id = param_pattern_inst->subpattern_id;
    inst = sem_ir.insts().Get(inst_id);
  }

  if (inst.Is<ReturnSlotPattern>()) {
    return NameId::ReturnSlot;
  }

  if (auto binding_pattern = inst.TryAs<AnyBindingPattern>()) {
    if (binding_pattern->entity_name_id.has_value()) {
      return sem_ir.entity_names().Get(binding_pattern->entity_name_id).name_id;
    } else {
      return NameId::Underscore;
    }
  }

  return NameId::None;
}

}  // namespace Carbon::SemIR

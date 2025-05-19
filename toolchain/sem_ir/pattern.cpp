// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/pattern.h"

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Returns the pattern instruction corresponding to the given ID, after
// unwrapping any simple pattern operators such as `var` and `addr`.
static auto GetUnwrapped(const File& sem_ir, InstId pattern_id)
    -> std::pair<InstId, Inst> {
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

  return {inst_id, inst};
}

// Returns the name and entity name introduced by the given instruction if it is
// a binding pattern, or otherwise `{None, None}`.
static auto GetBoundEntityName(const File& sem_ir, Inst inst)
    -> std::pair<NameId, EntityNameId> {
  if (auto binding_pattern = inst.TryAs<AnyBindingPattern>()) {
    return {sem_ir.entity_names().Get(binding_pattern->entity_name_id).name_id,
            binding_pattern->entity_name_id};
  }
  return {SemIR::NameId::None, SemIR::EntityNameId::None};
}

auto IsSelfPattern(const File& sem_ir, InstId pattern_id) -> bool {
  auto [_, inst] = GetUnwrapped(sem_ir, pattern_id);
  auto [name_id, entity_name_id] = GetBoundEntityName(sem_ir, inst);
  return name_id == NameId::SelfValue;
}

auto GetFirstBindingNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> EntityNameId {
  llvm::SmallVector<InstId> work_list = {pattern_id};
  while (!work_list.empty()) {
    auto [_, inst] = GetUnwrapped(sem_ir, work_list.pop_back_val());
    if (auto tuple_patt = inst.TryAs<SemIR::TuplePattern>()) {
      auto block = sem_ir.inst_blocks().Get(tuple_patt->elements_id);
      work_list.append(block.rbegin(), block.rend());
      continue;
    }

    // TODO: Look through struct patterns.

    auto [name_id, entity_name_id] = GetBoundEntityName(sem_ir, inst);
    CARBON_CHECK(entity_name_id.has_value(), "Unhandled pattern inst kind {0}",
                 inst);

    // Skip unnamed bindings.
    if (name_id != NameId::Underscore) {
      return entity_name_id;
    }
  }
  return EntityNameId::None;
}

auto GetPrettyNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> NameId {
  auto [inst_id, inst] = GetUnwrapped(sem_ir, pattern_id);

  if (auto [name_id, entity_name_id] = GetBoundEntityName(sem_ir, inst);
      entity_name_id.has_value()) {
    return name_id;
  }

  if (inst.Is<ReturnSlotPattern>()) {
    return NameId::ReturnSlot;
  }

  return NameId::None;
}

}  // namespace Carbon::SemIR

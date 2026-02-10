// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/constant.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst, ConstantDependence dependence)
    -> ConstantId {
  auto result = map_.Insert(inst, [&] {
    auto inst_id = sem_ir_->insts().AddInNoBlock(LocIdAndInst::NoLoc(inst));
    ConstantId const_id = ConstantId::None;
    if (dependence == ConstantDependence::None) {
      const_id = ConstantId::ForConcreteConstant(inst_id);
    } else {
      // The instruction in the constants store is an abstract symbolic
      // constant, not associated with any particular generic.
      SymbolicConstant symbolic_constant = {.inst_id = inst_id,
                                            .generic_id = GenericId::None,
                                            .index = GenericInstIndex::None,
                                            .dependence = dependence};
      const_id =
          sem_ir_->constant_values().AddSymbolicConstant(symbolic_constant);
    }
    sem_ir_->constant_values().Set(inst_id, const_id);
    constants_.push_back(inst_id);
    return const_id;
  });
  CARBON_CHECK(result.value() != ConstantId::None);
  CARBON_CHECK(
      result.value().is_symbolic() == (dependence != ConstantDependence::None),
      "Constant {0} registered as both symbolic and concrete constant.", inst);
  return result.value();
}

auto GetInstWithConstantValue(const File& file, ConstantId const_id) -> InstId {
  if (!const_id.has_value() || !const_id.is_constant()) {
    return InstId::None;
  }

  // For concrete constants, the corresponding instruction has the desired
  // constant value.
  if (!const_id.is_symbolic()) {
    return file.constant_values().GetInstId(const_id);
  }

  // For unattached symbolic constants, the corresponding instruction has the
  // desired constant value.
  const auto& symbolic_const =
      file.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic_const.generic_id.has_value()) {
    return file.constant_values().GetInstId(const_id);
  }

  // For attached symbolic constants, pick the corresponding instruction out of
  // the eval block for the generic.
  const auto& generic = file.generics().Get(symbolic_const.generic_id);
  auto block = generic.GetEvalBlock(symbolic_const.index.region());
  return file.inst_blocks().Get(block)[symbolic_const.index.index()];
}

}  // namespace Carbon::SemIR

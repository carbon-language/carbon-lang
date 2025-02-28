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
      const_id = SemIR::ConstantId::ForConcreteConstant(inst_id);
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

}  // namespace Carbon::SemIR

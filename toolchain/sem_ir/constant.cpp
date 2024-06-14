// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/constant.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId {
  auto [it, added] = map_.insert({inst, ConstantId::Invalid});
  if (added) {
    auto inst_id = sem_ir_.insts().AddInNoBlock(LocIdAndInst::NoLoc(inst));
    ConstantId const_id = ConstantId::Invalid;
    if (is_symbolic) {
      // The instruction in the constants store is an abstract symbolic
      // constant, not associated with any particular generic.
      auto symbolic_constant =
          SymbolicConstant{.inst_id = inst_id,
                           .generic_id = GenericId::Invalid,
                           .index = GenericInstIndex::Invalid};
      const_id =
          sem_ir_.constant_values().AddSymbolicConstant(symbolic_constant);
    } else {
      const_id = SemIR::ConstantId::ForTemplateConstant(inst_id);
    }
    it->second = const_id;
    sem_ir_.constant_values().Set(inst_id, const_id);
    constants_.push_back(inst_id);
  } else {
    CARBON_CHECK(it->second != ConstantId::Invalid);
    CARBON_CHECK(it->second.is_symbolic() == is_symbolic);
  }
  return it->second;
}

}  // namespace Carbon::SemIR

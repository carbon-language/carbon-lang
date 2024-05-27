// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/constant.h"

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId {
  auto result = map_.Insert(inst, [&] {
    auto inst_id = sem_ir_.insts().AddInNoBlock(LocIdAndInst::NoLoc(inst));
    auto const_id = is_symbolic
                        ? SemIR::ConstantId::ForSymbolicConstant(inst_id)
                        : SemIR::ConstantId::ForTemplateConstant(inst_id);
    sem_ir_.constant_values().Set(inst_id, const_id);
    constants_.push_back(inst_id);
    return const_id;
  });
  CARBON_CHECK(result.value() != ConstantId::Invalid);
  CARBON_CHECK(result.value().is_symbolic() == is_symbolic);
  return result.value();
}

}  // namespace Carbon::SemIR

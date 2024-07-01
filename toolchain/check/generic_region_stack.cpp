// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic_region_stack.h"

namespace Carbon::Check {

auto GenericRegionStack::Push() -> void {
  regions_.push_back(
      {.first_dependent_inst = static_cast<int32_t>(dependent_insts_.size())});
}

auto GenericRegionStack::Pop() -> void {
  auto region = regions_.pop_back_val();
  dependent_insts_.truncate(region.first_dependent_inst);
}

auto GenericRegionStack::AddDependentInst(DependentInst inst) -> void {
  CARBON_CHECK(!regions_.empty())
      << "Formed a dependent instruction while not in a generic region.";
  CARBON_CHECK(inst.kind != DependencyKind::None);
  dependent_insts_.push_back(inst);
}

auto GenericRegionStack::PeekDependentInsts() -> llvm::ArrayRef<DependentInst> {
  CARBON_CHECK(!regions_.empty());
  return llvm::ArrayRef(dependent_insts_)
      .slice(regions_.back().first_dependent_inst);
}

}  // namespace Carbon::Check

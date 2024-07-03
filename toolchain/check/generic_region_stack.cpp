// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic_region_stack.h"

namespace Carbon::Check {

auto GenericRegionStack::Push() -> void { dependent_insts_stack_.PushArray(); }

auto GenericRegionStack::Pop() -> void { dependent_insts_stack_.PopArray(); }

auto GenericRegionStack::AddDependentInst(DependentInst inst) -> void {
  dependent_insts_stack_.AppendToTop(inst);
}

auto GenericRegionStack::PeekDependentInsts() -> llvm::ArrayRef<DependentInst> {
  return dependent_insts_stack_.PeekArray();
}

}  // namespace Carbon::Check

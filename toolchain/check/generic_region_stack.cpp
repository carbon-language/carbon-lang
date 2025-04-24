// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic_region_stack.h"

namespace Carbon::Check {

auto GenericRegionStack::Push() -> void { dependent_insts_stack_.PushArray(); }

auto GenericRegionStack::Pop() -> void { dependent_insts_stack_.PopArray(); }

auto GenericRegionStack::AddDependentInst(DependentInst inst) -> void {
  if (dependent_insts_stack_.empty()) {
    // If we don't have a generic region here, leave the dependent instruction
    // unattached. This happens for out-of-line redeclarations of members of
    // dependent scopes:
    //
    //   class A(T:! type) {
    //     fn F();
    //   }
    //   // Has generic type and constant value, but no generic region.
    //   fn A(T:! type).F() {}
    //
    // TODO: Use a different instruction kind for out-of-line definitions and
    // CHECK this doesn't happen.
    return;
  }
  dependent_insts_stack_.AppendToTop(inst);
}

auto GenericRegionStack::PeekDependentInsts() -> llvm::ArrayRef<DependentInst> {
  return dependent_insts_stack_.PeekArray();
}

}  // namespace Carbon::Check

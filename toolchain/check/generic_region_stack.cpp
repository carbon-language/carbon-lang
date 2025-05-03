// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic_region_stack.h"

namespace Carbon::Check {

auto GenericRegionStack::Push(PendingGeneric generic) -> void {
  pending_generic_ids_.push_back(generic);
  pending_eval_block_stack_.PushArray();
  dependent_inst_stack_.PushArray();
  constants_in_generic_stack_.emplace_back();
}

auto GenericRegionStack::Pop() -> void {
  pending_generic_ids_.pop_back();
  pending_eval_block_stack_.PopArray();
  dependent_inst_stack_.PopArray();
  constants_in_generic_stack_.pop_back();
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern_block_stack.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto PatternBlockStack::Push() -> void { blocks_.PushArray(); }

auto PatternBlockStack::AddInst(SemIR::InstId inst_id) -> void {
  blocks_.AppendToTop(inst_id);
}

auto PatternBlockStack::Pop() -> SemIR::InstBlockId {
  auto block_id = context_->inst_blocks().Add(blocks_.PeekArray());
  blocks_.PopArray();
  return block_id;
}

}  // namespace Carbon::Check

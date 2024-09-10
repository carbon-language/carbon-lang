// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_PATTERN_BLOCK_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_PATTERN_BLOCK_STACK_H_

#include "common/array_stack.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

// Stack of blocks containing pattern insts.
class PatternBlockStack {
 public:
  explicit PatternBlockStack(Context* context) : context_(context) {}

  // Pushes a new pattern block. Each pattern block should correspond to a
  // single full pattern, i.e. a pattern that is not part of an enclosing
  // pattern.
  auto Push() -> void;

  // Adds inst_id to the current pattern block.
  auto AddInst(SemIR::InstId inst_id) -> void;

  // Finalizes and pops the top pattern block.
  [[nodiscard]] auto Pop() -> SemIR::InstBlockId;

 private:
  Context* context_;
  ArrayStack<SemIR::InstId> blocks_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_PATTERN_BLOCK_STACK_H_

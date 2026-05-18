// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GLOBAL_INIT_H_
#define CARBON_TOOLCHAIN_CHECK_GLOBAL_INIT_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

// Tracks state for global initialization. Handles should `Resume` when entering
// an expression that's used for global init, and `Suspend` when the expression
// is finished. Instructions in the middle will be tracked for the
// `__global_init` function.
class GlobalInit {
 public:
  explicit GlobalInit(Context* context) : context_(context) {}

  // Resumes adding instructions to global init.
  auto Resume() -> void;

  // Suspends adding instructions to global init.
  auto Suspend() -> void;

  // Finalizes the global initialization state, creating `__global_init` if
  // needed. Only called once at the end of checking.
  auto Finalize() -> void;

 private:
  // The associated context. Stored for convenience.
  Context* context_;

  // The currently suspended global init block. The value may change as a result
  // of control flow in initialization.
  SemIR::InstBlockId block_id_ = SemIR::InstBlockId::GlobalInit;

  // The contents for the currently suspended global init block.
  llvm::SmallVector<SemIR::InstId> block_;
};

// Stack that keeps track of whether to use global initialization for
// the initializer of a declaration.
class UseGlobalInitStack {
 public:
  // Pushes `use_global_init` onto the stack for a new declaration.
  auto Push(bool use_global_init) -> void;

  // Pops the stack.
  auto Pop() -> void;

  // Changes the value on the top of the stack to enabled.
  auto EnableUseGlobalInit() -> void;

  // Peeks the current value on the top of the stack.
  auto UseGlobalInit() -> bool;

 private:
  llvm::SmallVector<bool> stack_;
};

// Whether to use global initialization for the initializer of a declaration.
auto UseGlobalInit(Context& context) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GLOBAL_INIT_H_

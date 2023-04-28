// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/stack_space.h"

#include "common/check.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CrashRecoveryContext.h"

namespace Carbon::Internal {

static constexpr int64_t SufficientStack = 256 << 10;
static constexpr int64_t DesiredStackSpace = 8 << 20;

static LLVM_THREAD_LOCAL intptr_t bottom_of_stack = 0;

// Returns the current bottom of stack.
static auto GetStackPointer() -> intptr_t {
  char char_on_stack = 0;
  char* volatile ptr = &char_on_stack;
  return reinterpret_cast<intptr_t>(ptr);
}

auto IsStackSpaceNearlyExhausted() -> bool {
  CARBON_CHECK(bottom_of_stack != 0) << "InitStackSpace not called";
  return std::abs(GetStackPointer() - bottom_of_stack) >
         (DesiredStackSpace - SufficientStack);
}

auto RunWithStackSpaceHelper(std::function<void()> fn) -> void {
  llvm::CrashRecoveryContext context;
  context.RunSafelyOnThread(
      [&] {
        bottom_of_stack = GetStackPointer();
        fn();
      },
      DesiredStackSpace);
}

}  // namespace Carbon::Internal

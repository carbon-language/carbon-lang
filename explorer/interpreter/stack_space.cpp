// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/stack_space.h"

#include "common/check.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CrashRecoveryContext.h"

namespace Carbon {

namespace Internal {

static LLVM_THREAD_LOCAL intptr_t bottom_of_stack = 0;

// Returns the current bottom of stack.
static auto GetStackPointer() -> intptr_t {
  char char_on_stack = 0;
  char* volatile ptr = &char_on_stack;
  return reinterpret_cast<intptr_t>(ptr);
}

auto IsStackSpaceNearlyExhausted() -> bool {
  CARBON_CHECK(bottom_of_stack != 0) << "InitStackSpace not called";
  return std::abs(GetStackPointer() - bottom_of_stack) < 1e6;
}

auto RunWithStackSpaceHelper(std::function<void()> fn) -> void {
  llvm::CrashRecoveryContext context;
  static constexpr int DesiredStackSize = 1e8;
  context.RunSafelyOnThread(
      [&] {
        InitStackSpace();
        fn();
      },
      DesiredStackSize);
}

}  // namespace Internal

auto InitStackSpace() -> void {
  Internal::bottom_of_stack = Internal::GetStackPointer();
  CARBON_CHECK(Internal::bottom_of_stack != 0) << "Issue with GetStackPointer";
}

}  // namespace Carbon

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
LLVM_NO_SANITIZE("address")
LLVM_ATTRIBUTE_NOINLINE static auto GetStackPointer() -> intptr_t {
#if __GNUC__ || __has_builtin(__builtin_frame_address)
  return reinterpret_cast<intptr_t>(__builtin_frame_address(0));
#else
  char char_on_stack = 0;
  char* volatile ptr = &char_on_stack;
  return reinterpret_cast<intptr_t>(ptr);
#endif
}

auto IsStackSpaceNearlyExhausted() -> bool {
  if (bottom_of_stack == 0) {
    // Not initialized on the thread; always start a new thread.
    return true;
  }
  return std::abs(GetStackPointer() - bottom_of_stack) >
         (DesiredStackSpace - SufficientStack);
}

auto RunWithExtraStackHelper(llvm::function_ref<void()> fn) -> void {
  llvm::CrashRecoveryContext context;
  context.RunSafelyOnThread(
      [&] {
        bottom_of_stack = GetStackPointer();
        fn();
      },
      DesiredStackSpace);
}

}  // namespace Carbon::Internal

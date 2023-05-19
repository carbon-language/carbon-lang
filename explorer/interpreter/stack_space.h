// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_
#define CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

#include <optional>

#include "llvm/ADT/STLFunctionalExtras.h"

namespace Carbon {

namespace Internal {

// Returns true if a new thread should be started for more stack space.
auto IsStackSpaceNearlyExhausted() -> bool;

// Starts a thread to run the function.
auto ReserveStackAndRunHelper(llvm::function_ref<void()> fn) -> void;

}  // namespace Internal

// Runs a function. May start a new thread for the function if too much stack
// space has been consumed since the last time a thread was spawned. Always
// spawns a new thread when it didn't create the current thread.
//
// Usage:
//   return ReserveStackIfExhaustedAndRun<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto ReserveStackIfExhaustedAndRun(llvm::function_ref<ReturnType()> fn)
    -> ReturnType {
  if (Internal::IsStackSpaceNearlyExhausted()) {
    std::optional<ReturnType> result;
    Internal::ReserveStackAndRunHelper([&] { result = fn(); });
    return std::move(*result);
  } else {
    return fn();
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

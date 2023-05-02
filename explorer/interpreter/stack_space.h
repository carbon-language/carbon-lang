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

// Initializes stack space information for ReserveStackAndRun by creating a new
// thread, and runs the function on the new thread. This only needs to be called
// once within a given recursion.
//
// Usage:
//   return ReserveStackAndRun<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto ReserveStackAndRun(llvm::function_ref<ReturnType()> fn) -> ReturnType {
  std::optional<ReturnType> result;
  Internal::ReserveStackAndRunHelper([&] { result = fn(); });
  return std::move(*result);
}

// Runs a function. May start a new thread for the function if too much stack
// space has been consumed since the last time a thread was spawned (by either
// ReserveStackAndRun or ReserveStackIfExhaustedAndRun). Requires
// ReserveStackAndRun be used first within the recursion.
//
// Usage:
//   return ReserveStackIfExhaustedAndRun<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto ReserveStackIfExhaustedAndRun(llvm::function_ref<ReturnType()> fn)
    -> ReturnType {
  if (Internal::IsStackSpaceNearlyExhausted()) {
    return ReserveStackAndRun(fn);
  } else {
    return fn();
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

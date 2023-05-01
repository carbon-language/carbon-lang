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
auto RunWithStackSpaceHelper(llvm::function_ref<void()> fn) -> void;

}  // namespace Internal

// Initializes stack space information for RunWithStackSpace by creating a new
// thread, and runs the function on the new thread. This only needs to be called
// once within a given recursion.
//
// Usage:
//   return InitStackSpace<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto InitStackSpace(llvm::function_ref<ReturnType()> fn) -> ReturnType {
  std::optional<ReturnType> result;
  Internal::RunWithStackSpaceHelper([&] { result = fn(); });
  return std::move(*result);
}

// Runs a function. May start a new thread for the function if too much stack
// space has been consumed since the last time a thread was spawned (by either
// InitStackSpace or RunWithStackSpace). Require InitStackSpace be used first
// within the recursion.
//
// Usage:
//   return RunWithStackSpace<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto RunWithStackSpace(llvm::function_ref<ReturnType()> fn) -> ReturnType {
  if (Internal::IsStackSpaceNearlyExhausted()) {
    std::optional<ReturnType> result;
    Internal::RunWithStackSpaceHelper([&] { result = fn(); });
    return std::move(*result);
  } else {
    return fn();
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_
#define CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

#include <functional>
#include <optional>

namespace Carbon {

namespace Internal {

// Returns true if a new thread should be started for more stack space.
auto IsStackSpaceNearlyExhausted() -> bool;

// Starts a thread to run the function.
auto RunWithStackSpaceHelper(std::function<void()> fn) -> void;

}  // namespace Internal

// Starts a new thread with initialized stack space. This should be called
// before RunWithStackSpace so that execution is occurring with known stack
// space.
//
// Usage:
//   return InitStackSpace<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto InitStackSpace(std::function<ReturnType()> fn) -> ReturnType {
  std::optional<ReturnType> result;
  Internal::RunWithStackSpaceHelper([&] { result = fn(); });
  return std::move(*result);
}

// Runs a function. May start a thread if more stack space is desirable.
//
// Usage:
//   return StackSpaceRun<ReturnType>([&]() -> ReturnType {
//         <function body>
//       });
template <typename ReturnType>
auto RunWithStackSpace(std::function<ReturnType()> fn) -> ReturnType {
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

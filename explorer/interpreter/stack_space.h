// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_
#define CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

#include <functional>
#include <optional>

namespace Carbon {

namespace Internal {
auto IsStackSpaceNearlyExhausted() -> bool;
auto RunWithStackSpaceHelper(std::function<void()> fn) -> void;
}  // namespace Internal

auto InitStackSpace() -> void;

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

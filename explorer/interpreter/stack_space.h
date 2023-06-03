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
auto RunWithExtraStackHelper(llvm::function_ref<void()> fn) -> void;

}  // namespace Internal

// Runs `fn` after ensuring there is a reasonable amount of space left on the
// stack for it to run in. This will run `fn` in a separate thread if there is
// not enough space left on the current stack, or if RunWithExtraStack didn't
// create the current thread.
//
// Usage:
//   return RunWithExtraStack([&]() -> ReturnType {
//         <function body>
//       });
template <typename Fn>
auto RunWithExtraStack(Fn fn) -> decltype(fn()) {
  using ReturnType = decltype(fn());
  static_assert(!std::is_reference_v<ReturnType>);
  if (Internal::IsStackSpaceNearlyExhausted()) {
    std::optional<ReturnType> result;
    Internal::RunWithExtraStackHelper([&] { result = fn(); });
    return std::move(*result);
  } else {
    return fn();
  }
}

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_STACK_SPACE_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_NONNULL_H_
#define EXECUTABLE_SEMANTICS_COMMON_NONNULL_H_

#include <type_traits>

namespace Carbon {

// A non-nullable pointer. Written as `Nonnull<T*>` instead of `T*`.
//
// Sanitizers enforce this dynamically on assignment, return, and when passing
// as an argument. Static analysis will also track erroneous uses of `nullptr`.
template <typename T,
          typename std::enable_if_t<std::is_pointer_v<T>>* = nullptr>
using Nonnull = T _Nonnull;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_NONNULL_H_

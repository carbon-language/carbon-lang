// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CONCEPTS_H_
#define CARBON_COMMON_CONCEPTS_H_

#include <concepts>

namespace Carbon {

// True if `T` is the same as one of `OtherT`.
template <typename T, typename... OtherT>
concept SameAsOneOf = (std::same_as<T, OtherT> || ...);

}  // namespace Carbon

#endif  // CARBON_COMMON_CONCEPTS_H_

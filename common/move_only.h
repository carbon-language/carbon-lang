// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MOVE_ONLY_H_
#define CARBON_COMMON_MOVE_ONLY_H_

namespace Carbon {

// A base class that indicates a type is move-only. Typically this can be
// achieved by declaring the move constructor and move assignment yourself; this
// type should be used only when doing that is not feasible, such as when the
// type is an aggregate.
struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly&&) = default;
  auto operator=(MoveOnly&&) -> MoveOnly& = default;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_MOVE_ONLY_H_

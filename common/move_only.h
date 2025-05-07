// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MOVE_ONLY_H_
#define CARBON_COMMON_MOVE_ONLY_H_

namespace Carbon {

// A base class that indicates a type is move-only. Typically this can be
// achieved by declaring the move constructor and move assignment yourself; this
// type should be used only when doing that is not feasible, such as when
// aggregate initialization is still desired.
//
// This class uses CRTP to ensure that each MoveOnly base class has a different
// type. This is important to avoid the compiler adding extra padding to derived
// classes to give multiple MoveOnly subobjects of the same type different
// addresses.
template <typename Derived>
struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly&&) noexcept = default;
  auto operator=(MoveOnly&&) noexcept -> MoveOnly& = default;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_MOVE_ONLY_H_

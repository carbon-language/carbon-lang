// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_PARSER_PARSER_STACK_GUARD_H_
#define TOOLCHAIN_PARSER_PARSER_STACK_GUARD_H_

#include "common/check.h"

namespace Carbon {

class StackGuard {
 public:
  // This is meant to approximate stack limits, but we may need to find a better
  // way to track what the system is enforcing.
  static constexpr int Limit = 200;

  // Returns a root-level stack guard.
  static auto Root() -> StackGuard { return StackGuard(); }

  StackGuard(const StackGuard& parent) : depth_(parent.depth_ + 1) {
    CHECK(depth_ <= Limit) << "Exceeded recursion limit (" << Limit << ")";
  }

  auto is_at_limit() -> bool { return depth_ >= Limit; }

 private:
  StackGuard() = default;

  int depth_ = 0;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_PARSER_PARSER_STACK_GUARD_H_

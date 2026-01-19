// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_UNUSED_H_
#define CARBON_TOOLCHAIN_CHECK_UNUSED_H_

#include "toolchain/check/context.h"
#include "toolchain/check/scope_stack.h"

namespace Carbon::Check {

// Checks for unused bindings in the given scope.
auto CheckUnusedBindings(Context& context, ScopeStack::ScopeView scope) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_UNUSED_H_

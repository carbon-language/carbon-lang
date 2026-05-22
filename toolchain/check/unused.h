// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_UNUSED_H_
#define CARBON_TOOLCHAIN_CHECK_UNUSED_H_

#include "toolchain/check/context.h"
#include "toolchain/check/scope_stack.h"

namespace Carbon::Check {

// Checks for an unused binding. We track whether a name was declared in a
// reachable position, and whether it was used (even in unreachable code), to
// decide whether to warn. If a name is used in unreachable code, we don't warn
// because changing it to `_` would be a name lookup error in that unreachable
// code.
auto CheckUnusedBinding(Context& context, SemIR::NameId name_id,
                        const LexicalLookup::Result& result) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_UNUSED_H_

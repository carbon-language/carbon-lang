// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SUBPATTERN_H_
#define CARBON_TOOLCHAIN_CHECK_SUBPATTERN_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Marks the start of a region of insts in a pattern context that might
// represent an expression or a pattern. Typically this is called when
// handling a parse node that can immediately precede a subpattern (such
// as `let` or a `,` in a pattern list), and the handler for the subpattern
// node makes the matching `EndSubpatternAs*` call.
auto BeginSubpattern(Context& context) -> void;

// Ends a region started by BeginSubpattern (in stack order), treating it as
// an expression with the given result, and returns the ID of the region. The
// region will not yet have any control-flow edges into or out of it.
auto EndSubpatternAsExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId;

// Ends a region started by BeginSubpattern (in stack order), asserting that
// it had no expression content.
auto EndSubpatternAsNonExpr(Context& context) -> void;

// TODO: Add EndSubpatternAsPattern, when needed.

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SUBPATTERN_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_HANDLE_H_
#define CARBON_TOOLCHAIN_CHECK_HANDLE_H_

#include "toolchain/check/context.h"
#include "toolchain/check/function.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

// Parse node handlers. Returns false for unrecoverable errors.
#define CARBON_PARSE_NODE_KIND(Name) \
  auto Handle##Name(Context& context, Parse::Name##Id node_id) -> bool;
#include "toolchain/parse/node_kind.def"

// Handle suspending the definition of a function. This is used for inline
// methods, which are processed out of the normal lexical order. This plus
// HandleFunctionDefinitionResume carry out the same actions as
// HandleFunctionDefinitionStart, except that the various context stacks are
// cleared out in between.
auto HandleFunctionDefinitionSuspend(Context& context,
                                     Parse::FunctionDefinitionStartId node_id)
    -> SuspendedFunction;

// Handle resuming the definition of a function, after a previous suspension.
auto HandleFunctionDefinitionResume(Context& context,
                                    Parse::FunctionDefinitionStartId node_id,
                                    SuspendedFunction suspended_fn) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_HANDLE_H_

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_HANDLE_REQUIREMENT_H_
#define CARBON_TOOLCHAIN_PARSE_HANDLE_REQUIREMENT_H_

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Push state at the beginning of a requirement expression after a `where` or
// `require` token.
auto BeginRequirement(Context& context) -> void;

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_HANDLE_REQUIREMENT_H_

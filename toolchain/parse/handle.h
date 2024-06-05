// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_HANDLE_H_
#define CARBON_TOOLCHAIN_PARSE_HANDLE_H_

#include "toolchain/parse/context.h"

namespace Carbon::Parse {

// Declare handlers for each parse state.
#define CARBON_PARSE_STATE(Name) auto Handle##Name(Context& context) -> void;
#include "toolchain/parse/state.def"

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_HANDLE_H_

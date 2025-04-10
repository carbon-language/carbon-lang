// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/state.h"  // IWYU pragma: keep

namespace Carbon::Parse {

CARBON_DEFINE_ENUM_CLASS_NAMES(StateKind) = {
#define CARBON_PARSE_STATE(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/parse/state.def"
};

}  // namespace Carbon::Parse

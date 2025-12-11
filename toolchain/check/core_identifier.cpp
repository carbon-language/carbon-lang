// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/core_identifier.h"

namespace Carbon::Check {

CARBON_DEFINE_ENUM_CLASS_NAMES(CoreIdentifier) {
#define CARBON_CORE_IDENTIFIER(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
#include "toolchain/check/core_identifier.def"
};

}  // namespace Carbon::Check

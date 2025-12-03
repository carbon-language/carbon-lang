// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/keyword_modifier_set.h"

namespace Carbon::Check {

CARBON_DEFINE_ENUM_MASK_NAMES(KeywordModifierSet) {
  CARBON_KEYWORD_MODIFIER_SET(CARBON_ENUM_MASK_NAME_STRING)
};

}  // namespace Carbon::Check

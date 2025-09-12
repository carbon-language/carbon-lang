// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/node_category.h"

#include "llvm/ADT/StringExtras.h"

namespace Carbon::Parse {

CARBON_DEFINE_ENUM_MASK_NAMES(NodeCategory) = {
    CARBON_NODE_CATEGORY(CARBON_ENUM_MASK_NAME_STRING)};

}  // namespace Carbon::Parse

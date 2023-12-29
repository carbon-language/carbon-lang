// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleEmptyDecl(Context& /*context*/, Parse::EmptyDeclId /*parse_node*/)
    -> bool {
  // Empty declarations have no actions associated.
  return true;
}

auto HandleInvalidParse(Context& context, Parse::InvalidParseId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInvalidParse");
}

auto HandleInvalidParseStart(Context& context,
                             Parse::InvalidParseStartId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInvalidParseStart");
}

auto HandleInvalidParseSubtree(Context& context,
                               Parse::InvalidParseSubtreeId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInvalidParseSubtree");
}

auto HandlePlaceholder(Context& /*context*/,
                       Parse::PlaceholderId /*parse_node*/) -> bool {
  CARBON_FATAL()
      << "Placeholder node should always be replaced before parse completes";
}

}  // namespace Carbon::Check

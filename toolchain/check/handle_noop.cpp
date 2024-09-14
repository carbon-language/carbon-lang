// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/, Parse::EmptyDeclId /*node_id*/)
    -> bool {
  // Empty declarations have no actions associated.
  return true;
}

auto HandleParseNode(Context& context, Parse::InvalidParseId node_id) -> bool {
  return context.TODO(node_id, "HandleInvalidParse");
}

auto HandleParseNode(Context& context, Parse::InvalidParseStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleInvalidParseStart");
}

auto HandleParseNode(Context& context, Parse::InvalidParseSubtreeId node_id)
    -> bool {
  return context.TODO(node_id, "HandleInvalidParseSubtree");
}

auto HandleParseNode(Context& /*context*/, Parse::PlaceholderId /*node_id*/)
    -> bool {
  CARBON_FATAL(
      "Placeholder node should always be replaced before parse completes");
}

}  // namespace Carbon::Check

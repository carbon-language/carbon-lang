// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::PackExpansionStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandlePackExpansionStart");
}

auto HandleParseNode(Context& context, Parse::PackExpansionStatementId node_id)
    -> bool {
  return context.TODO(node_id, "HandlePackExpansionStatement");
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorExpandId node_id)
    -> bool {
  return context.TODO(node_id, "HandlePrefixOperatorExpand");
}

}  // namespace Carbon::Check

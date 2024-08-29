// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::SelfTypeNameId node_id) -> bool {
  return context.TODO(node_id, "HandleSelfTypeName");
}

auto HandleParseNode(Context& context, Parse::DesignatorExprId node_id)
    -> bool {
  return context.TODO(node_id, "HandleDesignatorExpr");
}

}  // namespace Carbon::Check

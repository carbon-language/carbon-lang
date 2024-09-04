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

auto HandleParseNode(Context& context, Parse::WhereIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleWhereIntroducer");
}

auto HandleParseNode(Context& context, Parse::RequirementAssignId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequirementAssign");
}

auto HandleParseNode(Context& context, Parse::RequirementEqualsId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequirementEquals");
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequirementImpls");
}

auto HandleParseNode(Context& context, Parse::WhereExprId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereExpr");
}

}  // namespace Carbon::Check

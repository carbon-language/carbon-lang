// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleWhereIntroducer(Context& context, Parse::WhereIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleWhereIntroducer");
}

auto HandleDotSelf(Context& context, Parse::DotSelfId node_id) -> bool {
  return context.TODO(node_id, "HandleDotSelf");
}

auto HandleRequirementAssign(Context& context,
                             Parse::RequirementAssignId node_id) -> bool {
  return context.TODO(node_id, "HandleRequirementAssign");
}

auto HandleRequirementEquals(Context& context,
                             Parse::RequirementEqualsId node_id) -> bool {
  return context.TODO(node_id, "HandleRequirementEquals");
}

auto HandleRequirementImpls(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequirementImpls");
}

auto HandleRequirementAnd(Context& context, Parse::RequirementAndId node_id)
    -> bool {
  return context.TODO(node_id, "HandleRequirementAnd");
}

auto HandleWhereExpr(Context& context, Parse::WhereExprId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereExpr");
}

}  // namespace Carbon::Check

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

auto HandleWhereAssign(Context& context, Parse::WhereAssignId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereAssign");
}

auto HandleWhereEquals(Context& context, Parse::WhereEqualsId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereEquals");
}

auto HandleWhereImpls(Context& context, Parse::WhereImplsId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereImpls");
}

auto HandleWhereAnd(Context& context, Parse::WhereAndId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereAnd");
}

auto HandleWhereExpr(Context& context, Parse::WhereExprId node_id) -> bool {
  return context.TODO(node_id, "HandleWhereExpr");
}

}  // namespace Carbon::Check

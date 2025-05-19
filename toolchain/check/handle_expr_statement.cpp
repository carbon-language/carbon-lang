// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ExprStatementId /*node_id*/)
    -> bool {
  DiscardExpr(context, context.node_stack().PopExpr());
  return true;
}

}  // namespace Carbon::Check

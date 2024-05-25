// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleCallExprStart(Context& context, Parse::CallExprStartId node_id)
    -> bool {
  auto name_id = context.node_stack().PopExpr();
  context.node_stack().Push(node_id, name_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleCallExprComma(Context& context, Parse::CallExprCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleCallExpr(Context& context, Parse::CallExprId node_id) -> bool {
  // Process the final explicit call argument now, but leave the arguments
  // block on the stack until the end of this function.
  context.param_and_arg_refs_stack().EndNoPop(Parse::NodeKind::CallExprStart);
  auto [call_expr_node_id, callee_id] =
      context.node_stack().PopWithNodeId<Parse::NodeKind::CallExprStart>();

  auto call_id = PerformCall(
      context, call_expr_node_id, callee_id,
      context.param_and_arg_refs_stack().PeekCurrentBlockContents());

  context.param_and_arg_refs_stack().PopAndDiscard();
  context.node_stack().Push(node_id, call_id);
  return true;
}

}  // namespace Carbon::Check

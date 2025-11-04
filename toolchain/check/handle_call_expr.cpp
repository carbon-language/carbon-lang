// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::CallExprStartId node_id) -> bool {
  auto name_id = context.node_stack().PopExpr();
  context.node_stack().Push(node_id, name_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::CallExprId node_id) -> bool {
  // Process the final explicit call argument now, but leave the arguments
  // block on the stack until the end of this function.
  context.param_and_arg_refs_stack().EndNoPop(Parse::NodeKind::CallExprStart);
  auto callee_id = context.node_stack().Pop<Parse::NodeKind::CallExprStart>();

  auto call_id = PerformCall(
      context, node_id, callee_id,
      context.param_and_arg_refs_stack().PeekCurrentBlockContents());

  context.param_and_arg_refs_stack().PopAndDiscard();
  context.node_stack().Push(node_id, call_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RefTagId node_id) -> bool {
  auto expr_id = context.node_stack().Peek<Parse::NodeCategory::Expr>();

  if (SemIR::GetExprCategory(context.sem_ir(), expr_id) !=
      SemIR::ExprCategory::DurableRef) {
    CARBON_DIAGNOSTIC(
        RefTagNotDurableRef, Error,
        "expression tagged with `ref` is not a durable reference");
    context.emitter().Emit(node_id, RefTagNotDurableRef);
  }
  context.ref_tags().Insert(expr_id, Context::RefTag::Present);
  return true;
}

}  // namespace Carbon::Check

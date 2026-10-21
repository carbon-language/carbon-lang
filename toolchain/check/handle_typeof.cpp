// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/,
                     Parse::TypeOfExprKeywordId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::TypeOfExprOpenParenId /*node_id*/)
    -> bool {
  // The operand of `typeof` is never evaluated at runtime, so build it in a
  // separate expression region that is not part of the enclosing control flow.
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  // Any cleanups created by the operand are never needed.
  context.scope_stack().PushForSameRegion(ScopeStack::CleanupScopeKind::None);
  return true;
}

auto HandleParseNode(Context& context, Parse::TypeOfExprId node_id) -> bool {
  auto operand_id = context.node_stack().PopExpr();
  context.scope_stack().Pop();

  // Finish building the operand region. Unlike regions that are later spliced
  // into a control flow graph, this region has no successor, so we don't add a
  // branch out of it.
  auto block_id = context.inst_block_stack().Pop();
  CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());
  auto operand_region_id = context.sem_ir().expr_regions().Add(
      {.block_ids = context.region_stack().PopRegion(),
       .result_id = operand_id});

  auto inst_id =
      AddInst<SemIR::TypeOf>(context, node_id,
                             {.type_id = SemIR::TypeType::TypeId,
                              .operand_region_id = operand_region_id});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

}  // namespace Carbon::Check

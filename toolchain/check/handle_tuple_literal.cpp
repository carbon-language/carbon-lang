// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleTupleLiteralStart(Context& context,
                             Parse::TupleLiteralStartId node_id) -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleTupleLiteralComma(Context& context,
                             Parse::TupleLiteralCommaId /*node_id*/) -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleTupleLiteral(Context& context, Parse::TupleLiteralId node_id)
    -> bool {
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::TupleLiteralStart);

  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::TupleLiteralStart>();
  const auto& inst_block = context.inst_blocks().Get(refs_id);
  llvm::SmallVector<SemIR::TypeId> type_ids;
  type_ids.reserve(inst_block.size());
  for (auto inst : inst_block) {
    type_ids.push_back(context.insts().Get(inst).type_id());
  }
  auto type_id = context.GetTupleType(type_ids);

  auto value_id =
      context.AddInst({node_id, SemIR::TupleLiteral{type_id, refs_id}});
  context.node_stack().Push(node_id, value_id);
  return true;
}

}  // namespace Carbon::Check

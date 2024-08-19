// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::ImplicitParamListStart);
  context.node_stack().Push(node_id, refs_id);
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::TuplePatternStart);
  // TODO: do this at full-pattern level.
  llvm::SmallVector<SemIR::InstId> inner_param_insts;
  auto refs_block = context.inst_blocks().Get(refs_id);
  inner_param_insts.reserve(refs_block.size());
  for (SemIR::InstId inst_id : refs_block) {
    // TODO: generalize for other pattern kinds.
    auto binding_pattern =
        context.insts().GetAs<SemIR::BindingPattern>(inst_id);
    context.inst_block_stack().AddInstId(binding_pattern.bind_inst_id);
    inner_param_insts.push_back(binding_pattern.bind_inst_id);
  }
  auto inner_param_block = context.inst_blocks().Add(inner_param_insts);
  context.node_stack().Push(node_id, inner_param_block);
  return true;
}

}  // namespace Carbon::Check

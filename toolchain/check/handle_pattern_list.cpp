// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.params_stack().Push();
  context.param_patterns_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  if (!context.node_stack().PeekIs(Parse::NodeKind::ImplicitParamListStart)) {
    auto [node_id, inst_id] = context.node_stack().PopPatternWithNodeId();
    context.params_stack().AddInstId(inst_id);
    context.param_patterns_stack().AddInstId(
        context.pattern_node_stack().Pop<SemIR::InstId>(node_id));
  }
  context.node_stack().Push(node_id, context.params_stack().Pop());
  context.pattern_node_stack().Push(node_id,
                                    context.param_patterns_stack().Pop());
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.params_stack().Push();
  context.param_patterns_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  auto [node_id, inst_id] = context.node_stack().PopPatternWithNodeId();
  context.params_stack().AddInstId(inst_id);
  context.param_patterns_stack().AddInstId(
      context.pattern_node_stack().Pop<SemIR::InstId>(node_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  if (!context.node_stack().PeekIs(Parse::NodeKind::TuplePatternStart)) {
    auto [node_id, inst_id] = context.node_stack().PopPatternWithNodeId();
    context.params_stack().AddInstId(inst_id);
    context.param_patterns_stack().AddInstId(
        context.pattern_node_stack().Pop<SemIR::InstId>(node_id));
  }
  context.node_stack().Push(node_id, context.params_stack().Pop());
  context.pattern_node_stack().Push(node_id,
                                    context.param_patterns_stack().Pop());
  return true;
}

}  // namespace Carbon::Check

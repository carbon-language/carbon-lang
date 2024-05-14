// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImplicitParamListStart(Context& context,
                                  Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleImplicitParamList(Context& context,
                             Parse::ImplicitParamListId node_id) -> bool {
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::ImplicitParamListStart);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
  context.node_stack().Push(node_id, refs_id);
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleTuplePatternStart(Context& context,
                             Parse::TuplePatternStartId node_id) -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandlePatternListComma(Context& context,
                            Parse::PatternListCommaId /*node_id*/) -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  return true;
}

auto HandleTuplePattern(Context& context, Parse::TuplePatternId node_id)
    -> bool {
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::TuplePatternStart);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::TuplePatternStart>();
  context.node_stack().Push(node_id, refs_id);
  return true;
}

}  // namespace Carbon::Check

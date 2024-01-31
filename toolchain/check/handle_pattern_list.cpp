// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImplicitParamListStart(Context& context,
                                  Parse::ImplicitParamListStartId parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandleImplicitParamList(Context& context,
                             Parse::ImplicitParamListId parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::ImplicitParamListStart);
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ImplicitParamListStart>();
  context.node_stack().Push(parse_node, refs_id);
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleTuplePatternStart(Context& context,
                             Parse::TuplePatternStartId parse_node) -> bool {
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandlePatternListComma(Context& context,
                            Parse::PatternListCommaId /*parse_node*/) -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleTuplePattern(Context& context, Parse::TuplePatternId parse_node)
    -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::TuplePatternStart);
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::TuplePatternStart>();
  context.node_stack().Push(parse_node, refs_id);
  return true;
}

}  // namespace Carbon::Check

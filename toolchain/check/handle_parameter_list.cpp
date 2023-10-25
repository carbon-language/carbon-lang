// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImplicitParameterList(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplicitParameterList");
}

auto HandleImplicitParameterListStart(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleImplicitParameterListStart");
}

auto HandleParameterList(Context& context, Parse::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::ParameterListStart);
  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ParameterListStart>();
  context.node_stack().Push(parse_node, refs_id);
  return true;
}

auto HandleParameterListComma(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleParameterListStart(Context& context, Parse::Node parse_node)
    -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleDeducedParameterList(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleDeducedParameterList");
}

auto SemanticsHandleDeducedParameterListStart(SemanticsContext& context,
                                              ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleDeducedParameterListStart");
}

auto SemanticsHandleParameterList(SemanticsContext& context,
                                  ParseTree::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::ParameterListStart);
  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<ParseNodeKind::ParameterListStart>();
  context.node_stack().Push(parse_node, refs_id);
  return true;
}

auto SemanticsHandleParameterListComma(SemanticsContext& context,
                                       ParseTree::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma(/*for_args=*/false);
  return true;
}

auto SemanticsHandleParameterListStart(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsHandleCodeBlockStart(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  // TODO: Enter a new scope.
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleCodeBlock(SemanticsContext& context,
                              ParseTree::Node /*parse_node*/) -> bool {
  // TODO: Leave the scope.
  context.node_stack().PopForSoloParseNode(ParseNodeKind::CodeBlockStart);
  return true;
}

}  // namespace Carbon

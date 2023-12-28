// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleCodeBlockStart(Context& context, Parse::CodeBlockStartId parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  context.PushScope();
  return true;
}

auto HandleCodeBlock(Context& context, Parse::CodeBlockId /*parse_node*/)
    -> bool {
  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::CodeBlockStart>();
  return true;
}

}  // namespace Carbon::Check

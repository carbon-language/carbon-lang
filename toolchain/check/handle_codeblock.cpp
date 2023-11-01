// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleCodeBlockStart(Context& context, Parse::Node parse_node) -> bool {
  context.lamp_stack().Push(parse_node);
  context.PushScope();
  return true;
}

auto HandleCodeBlock(Context& context, Parse::Node /*parse_node*/) -> bool {
  context.PopScope();
  context.lamp_stack().PopForSoloParseNode<Parse::NodeKind::CodeBlockStart>();
  return true;
}

}  // namespace Carbon::Check

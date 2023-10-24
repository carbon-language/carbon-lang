// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleFileStart(Context& context, Parse::Node parse_node) -> bool {
  // Push the file as a sentinel so that it's always safe to peek at the
  // enclosing node on the node stack to determine what context we're in.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleFileEnd(Context& context, Parse::Node /*parse_node*/) -> bool {
  context.node_stack().PopForSoloParseNode<Parse::NodeKind::FileStart>();
  return true;
}

}  // namespace Carbon::Check

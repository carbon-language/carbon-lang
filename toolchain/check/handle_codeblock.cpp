// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::CodeBlockStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.scope_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::CodeBlockId /*node_id*/) -> bool {
  context.scope_stack().Pop();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::CodeBlockStart>();
  return true;
}

}  // namespace Carbon::Check

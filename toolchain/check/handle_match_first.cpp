// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/,
                     Parse::MatchFirstIntroducerId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::MatchFirstDefinitionStartId node_id) -> bool {
  auto enclosing_scope_inst_id = context.scope_stack().PeekInstId();
  auto decl_id = AddInst<SemIR::MatchFirstDecl>(
      context, node_id, {.enclosing_scope_inst_id = enclosing_scope_inst_id});
  context.scope_stack().PushForMatchFirstBlock(decl_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::MatchFirstId /*node_id*/)
    -> bool {
  context.scope_stack().Pop();
  return true;
}

}  // namespace Carbon::Check

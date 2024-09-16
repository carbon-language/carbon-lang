// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto blocks = context.param_and_arg_refs_stack().EndAndPopWithPattern(
      Parse::NodeKind::ImplicitParamListStart);
  context.node_stack().Push(node_id, blocks.param_block);
  context.pattern_node_stack().Push(node_id, blocks.pattern_block);
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyCommaInPattern();
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto blocks = context.param_and_arg_refs_stack().EndAndPopWithPattern(
      Parse::NodeKind::TuplePatternStart);
  context.node_stack().Push(node_id, blocks.param_block);
  context.pattern_node_stack().Push(node_id, blocks.pattern_block);
  return true;
}

}  // namespace Carbon::Check

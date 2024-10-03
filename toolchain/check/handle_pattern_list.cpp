// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_patterns_stack().Push();
  return true;
}

static auto HandleTrailingParam(Context& context) {
  context.param_patterns_stack().AddInstId(context.node_stack().PopPattern());
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  if (!context.node_stack().PeekIs(Parse::NodeKind::ImplicitParamListStart)) {
    HandleTrailingParam(context);
  }
  context.node_stack().Push(node_id, context.param_patterns_stack().Pop());
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_patterns_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  HandleTrailingParam(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  if (!context.node_stack().PeekIs(Parse::NodeKind::TuplePatternStart)) {
    HandleTrailingParam(context);
  }
  context.node_stack().Push(node_id, context.param_patterns_stack().Pop());
  return true;
}

}  // namespace Carbon::Check

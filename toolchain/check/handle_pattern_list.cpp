// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleImplicitParamList(Context& context, Parse::NodeId parse_node)
    -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::ImplicitParamListStart);
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ImplicitParamListStart>();
  context.node_stack().Push(parse_node, refs_id);
  // The implicit parameter list's scope extends to the end of the following
  // parameter list.
  return true;
}

auto HandleImplicitParamListStart(Context& context, Parse::NodeId parse_node)
    -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandleTuplePattern(Context& context, Parse::NodeId parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::TuplePatternStart);
  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::TuplePatternStart>();
  context.node_stack().Push(parse_node, refs_id);
  return true;
}

auto HandlePatternListComma(Context& context, Parse::NodeId /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleTuplePatternStart(Context& context, Parse::NodeId parse_node)
    -> bool {
  // A tuple pattern following an implicit parameter list shares the same
  // scope.
  //
  // TODO: For a declaration like
  //
  //   fn A(T:! type).B(U:! T).C(x: X(U)) { ... }
  //
  // ... all the earlier parameter should be in scope in the later parameter
  // lists too.
  if (!context.node_stack().PeekIs<Parse::NodeKind::ImplicitParamList>()) {
    context.PushScope();
  }
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon::Check

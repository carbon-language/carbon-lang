// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleCallExpression(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  auto [ir_id, refs_id] = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::CallExpressionStart);

  // TODO: Convert to call expression.
  auto [call_expr_parse_node, name_id] =
      context.node_stack().PopForParseNodeAndNodeId(
          ParseNodeKind::CallExpressionStart);
  auto name_node = context.semantics().GetNode(name_id);
  if (name_node.kind() != SemanticsNodeKind::FunctionDeclaration) {
    // TODO: Work on error.
    context.TODO(parse_node, "Not a callable name");
    context.node_stack().Push(parse_node, name_id);
    return true;
  }

  auto [_, callable_id] = name_node.GetAsFunctionDeclaration();
  auto callable = context.semantics().GetCallable(callable_id);

  CARBON_DIAGNOSTIC(NoMatchingCall, Error, "No matching callable was found.");
  auto diagnostic =
      context.emitter().Build(call_expr_parse_node, NoMatchingCall);
  if (!context.ImplicitAsForArgs(ir_id, refs_id, name_node.parse_node(),
                                 callable.param_refs_id, &diagnostic)) {
    diagnostic.Emit();
    context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinInvalidType);
    return true;
  }

  CARBON_CHECK(context.ImplicitAsForArgs(ir_id, refs_id, name_node.parse_node(),
                                         callable.param_refs_id,
                                         /*diagnostic=*/nullptr));

  auto call_id = context.semantics().AddCall({ir_id, refs_id});
  // TODO: Propagate return types from callable.
  auto call_node_id = context.AddNode(SemanticsNode::Call::Make(
      call_expr_parse_node, callable.return_type_id, call_id, callable_id));

  context.node_stack().Push(parse_node, call_node_id);
  return true;
}

auto SemanticsHandleCallExpressionComma(SemanticsContext& context,
                                        ParseTree::Node /*parse_node*/)
    -> bool {
  context.ParamOrArgComma(/*for_args=*/true);
  return true;
}

auto SemanticsHandleCallExpressionStart(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  auto name_id =
      context.node_stack().PopForNodeId(ParseNodeKind::NameReference);
  context.node_stack().Push(parse_node, name_id);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon

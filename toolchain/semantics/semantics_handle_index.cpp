// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsHandleIndexExpressionStart(SemanticsContext& context,
                                         ParseTree::Node parse_node) -> bool {
  context.node_stack().Push(parse_node, context.node_stack().PopExpression());
  return true;
}

auto SemanticsHandleIndexExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto ind_node = context.node_stack().PopExpression();
  auto [ind_expr_parse_node, name_id] =
      context.node_stack()
          .PopWithParseNode<ParseNodeKind::IndexExpressionStart>();
  auto name_node = context.semantics_ir().GetNode(name_id);
  auto name_type = context.semantics_ir().GetNode(
      context.semantics_ir().GetType(name_node.type_id()));
  auto cast_ind_id = context.ImplicitAsRequired(
      parse_node, ind_node,
      context.CanonicalizeType(SemanticsNodeId::BuiltinIntegerType));

  if (cast_ind_id.is_valid() &&
      name_type.kind() == SemanticsNodeKind::TupleType) {
    auto cast_ind_node = context.semantics_ir().GetNode(cast_ind_id);
    auto ind_val = context.semantics_ir()
                       .GetIntegerLiteral(cast_ind_node.GetAsIntegerLiteral())
                       .getSExtValue();
    if (ind_val >> 31) {
      context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
      return true;
    }
    auto type_block_id = name_type.GetAsTupleType();
    auto type_block = context.semantics_ir().GetTypeBlock(type_block_id);
    context.node_stack().Push(parse_node,
                              context.AddNode(SemanticsNode::Index::Make(
                                  parse_node, type_block[ind_val], name_id)));
  } else {
    context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  }
  return true;
}

}  // namespace Carbon

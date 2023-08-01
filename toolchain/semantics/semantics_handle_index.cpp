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
  auto [ind_expr_parse_node, name_node_id] =
      context.node_stack()
          .PopWithParseNode<ParseNodeKind::IndexExpressionStart>();
  auto name_node = context.semantics_ir().GetNode(name_node_id);

  if (context.semantics_ir().GetTypeAllowBuiltinTypes(name_node.type_id()) !=
      SemanticsNodeId::BuiltinError) {
    auto name_type = context.semantics_ir().GetNode(
        context.semantics_ir().GetType(name_node.type_id()));
    auto cast_ind_id = context.ImplicitAsRequired(
        parse_node, ind_node,
        context.CanonicalizeType(SemanticsNodeId::BuiltinIntegerType));
    auto cast_ind_node = context.semantics_ir().GetNode(cast_ind_id);

    if ((context.semantics_ir().GetTypeAllowBuiltinTypes(
             cast_ind_node.type_id()) != SemanticsNodeId::BuiltinError) &&
        (name_type.kind() == SemanticsNodeKind::TupleType)) {
      auto ind_val = context.semantics_ir()
                         .GetIntegerLiteral(cast_ind_node.GetAsIntegerLiteral())
                         .getSExtValue();
      auto type_block_id = name_type.GetAsTupleType();
      auto type_block = context.semantics_ir().GetTypeBlock(type_block_id);
      if (ind_val >= static_cast<int>(type_block.size())) {
        CARBON_DIAGNOSTIC(OutOfBoundAccess, Error, "Out of bound access.");
        context.emitter().Emit(parse_node, OutOfBoundAccess);
      } else {
        context.node_stack().Push(
            parse_node,
            context.AddNode(SemanticsNode::Index::Make(
                parse_node, type_block[ind_val], name_node_id, cast_ind_id)));
        return true;
      }
    }
  }
  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon

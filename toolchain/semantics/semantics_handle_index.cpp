// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsHandleIndexExpressionStart(SemanticsContext& /*context*/,
                                         ParseTree::Node /*parse_node*/)
    -> bool {
  // Leave the expression on the stack for IndexExpression.
  return true;
}

auto SemanticsHandleIndexExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto ind_node_id = context.node_stack().PopExpression();
  auto name_node_id = context.node_stack().PopExpression();
  auto name_node = context.semantics_ir().GetNode(name_node_id);

  if (context.semantics_ir().GetNode(ind_node_id).kind() ==
      SemanticsNodeKind::VarStorage) {
    CARBON_DIAGNOSTIC(NondeterministicType, Error,
                      "Type cannot be determined in compile time.");
    context.emitter().Emit(parse_node, NondeterministicType);
  } else if (name_node.kind() != SemanticsNodeKind::VarStorage) {
    CARBON_DIAGNOSTIC(InvalidIndexExpression, Error,
                      "Invalid index expression.");
    context.emitter().Emit(parse_node, InvalidIndexExpression);
  } else {
    auto name_type = context.semantics_ir().GetNode(
        context.semantics_ir().GetType(name_node.type_id()));
    auto cast_ind_id = context.ImplicitAsRequired(
        parse_node, ind_node_id,
        context.CanonicalizeType(SemanticsNodeId::BuiltinIntegerType));
    auto cast_ind_node = context.semantics_ir().GetNode(cast_ind_id);

    if (cast_ind_node.type_id() != SemanticsTypeId::Error &&
        name_type.kind() == SemanticsNodeKind::TupleType) {
      auto ind_val = context.semantics_ir()
                         .GetIntegerLiteral(cast_ind_node.GetAsIntegerLiteral())
                         .getSExtValue();
      auto type_block_id = name_type.GetAsTupleType();
      auto type_block = context.semantics_ir().GetTypeBlock(type_block_id);
      if (ind_val >= static_cast<int>(type_block.size())) {
        CARBON_DIAGNOSTIC(OutOfBoundsAccess, Error,
                          "Index: {0} is out of bound {1}.", int64_t, int64_t);
        context.emitter().Emit(parse_node, OutOfBoundsAccess, ind_val,
                               static_cast<int>(type_block.size()));
      } else {
        context.AddNodeAndPush(parse_node, SemanticsNode::Index::Make(
                                               parse_node, type_block[ind_val],
                                               name_node_id, cast_ind_id));
        return true;
      }
    }
  }

  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APSInt.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Check {

auto HandleIndexExpressionStart(Context& /*context*/,
                                Parse::Node /*parse_node*/) -> bool {
  // Leave the expression on the stack for IndexExpression.
  return true;
}

// Validates that the index (required to be an IntegerLiteral) is valid within
// the array or tuple size. Returns the index on success, or nullptr on failure.
static auto ValidateIntegerLiteralBound(Context& context,
                                        Parse::Node parse_node,
                                        SemIR::Node operand_node,
                                        SemIR::Node index_node, int size)
    -> const llvm::APInt* {
  const auto& index_val = context.semantics_ir().GetIntegerLiteral(
      index_node.GetAsIntegerLiteral());
  if (index_val.uge(size)) {
    CARBON_DIAGNOSTIC(IndexOutOfBounds, Error,
                      "Index `{0}` is past the end of `{1}`.", llvm::APSInt,
                      std::string);
    context.emitter().Emit(
        parse_node, IndexOutOfBounds,
        llvm::APSInt(index_val, /*isUnsigned=*/true),
        context.semantics_ir().StringifyType(operand_node.type_id()));
    return nullptr;
  }
  return &index_val;
}

auto HandleIndexExpression(Context& context, Parse::Node parse_node) -> bool {
  auto index_node_id = context.node_stack().PopExpression();
  auto index_node = context.semantics_ir().GetNode(index_node_id);
  auto operand_node_id = context.node_stack().PopExpression();
  operand_node_id = context.MaterializeIfInitializing(operand_node_id);
  auto operand_node = context.semantics_ir().GetNode(operand_node_id);
  auto operand_type_id = operand_node.type_id();
  auto operand_type_node = context.semantics_ir().GetNode(
      context.semantics_ir().GetTypeAllowBuiltinTypes(operand_type_id));

  switch (operand_type_node.kind()) {
    case SemIR::NodeKind::ArrayType: {
      auto [bound_id, element_type_id] = operand_type_node.GetAsArrayType();
      // We can check whether integers are in-bounds, although it doesn't affect
      // the IR for an array.
      if (index_node.kind() == SemIR::NodeKind::IntegerLiteral &&
          !ValidateIntegerLiteralBound(
              context, parse_node, operand_node, index_node,
              context.semantics_ir().GetArrayBoundValue(bound_id))) {
        index_node_id = SemIR::NodeId::BuiltinError;
      }
      auto cast_index_id = context.ConvertToValueOfType(
          index_node.parse_node(), index_node_id,
          context.CanonicalizeType(SemIR::NodeId::BuiltinIntegerType));
      context.AddNodeAndPush(parse_node, SemIR::Node::ArrayIndex::Make(
                                             parse_node, element_type_id,
                                             operand_node_id, cast_index_id));
      return true;
    }
    case SemIR::NodeKind::TupleType: {
      SemIR::TypeId element_type_id = SemIR::TypeId::Error;
      if (index_node.kind() == SemIR::NodeKind::IntegerLiteral) {
        auto type_block = context.semantics_ir().GetTypeBlock(
            operand_type_node.GetAsTupleType());
        if (const auto* index_val =
                ValidateIntegerLiteralBound(context, parse_node, operand_node,
                                            index_node, type_block.size())) {
          element_type_id = type_block[index_val->getZExtValue()];
        } else {
          index_node_id = SemIR::NodeId::BuiltinError;
        }
      } else {
        CARBON_DIAGNOSTIC(TupleIndexIntegerLiteral, Error,
                          "Tuples indices must be integer literals.");
        context.emitter().Emit(parse_node, TupleIndexIntegerLiteral);
        index_node_id = SemIR::NodeId::BuiltinError;
      }
      context.AddNodeAndPush(parse_node, SemIR::Node::TupleIndex::Make(
                                             parse_node, element_type_id,
                                             operand_node_id, index_node_id));
      return true;
    }
    default: {
      if (operand_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TypeNotIndexable, Error,
                          "`{0}` does not support indexing.", std::string);
        context.emitter().Emit(
            parse_node, TypeNotIndexable,
            context.semantics_ir().StringifyType(operand_type_id));
      }
      context.node_stack().Push(parse_node, SemIR::NodeId::BuiltinError);
      return true;
    }
  }
}

}  // namespace Carbon::Check

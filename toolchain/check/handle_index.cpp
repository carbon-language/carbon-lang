// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APSInt.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
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
                                        SemIR::IntegerLiteral index_node,
                                        int size) -> const llvm::APInt* {
  const auto& index_val =
      context.semantics_ir().GetInteger(index_node.integer_id);
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
  operand_node_id =
      ConvertToValueOrReferenceExpression(context, operand_node_id);
  auto operand_node = context.semantics_ir().GetNode(operand_node_id);
  auto operand_type_id = operand_node.type_id();
  auto operand_type_node = context.semantics_ir().GetNode(
      context.semantics_ir().GetTypeAllowBuiltinTypes(operand_type_id));

  switch (operand_type_node.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = operand_type_node.As<SemIR::ArrayType>();
      // We can check whether integers are in-bounds, although it doesn't affect
      // the IR for an array.
      if (auto index_literal = index_node.TryAs<SemIR::IntegerLiteral>();
          index_literal &&
          !ValidateIntegerLiteralBound(
              context, parse_node, operand_node, *index_literal,
              context.semantics_ir().GetArrayBoundValue(array_type.bound_id))) {
        index_node_id = SemIR::NodeId::BuiltinError;
      }
      auto cast_index_id = ConvertToValueOfType(
          context, index_node.parse_node(), index_node_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntegerType));
      auto array_cat =
          SemIR::GetExpressionCategory(context.semantics_ir(), operand_node_id);
      if (array_cat == SemIR::ExpressionCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_node_id = context.AddNode(SemIR::ValueAsReference(
            parse_node, operand_type_id, operand_node_id));
      }
      auto elem_id = context.AddNode(
          SemIR::ArrayIndex(parse_node, array_type.element_type_id,
                            operand_node_id, cast_index_id));
      if (array_cat != SemIR::ExpressionCategory::DurableReference) {
        // Indexing a durable reference gives a durable reference expression.
        // Indexing anything else gives a value expression.
        // TODO: This should be replaced by a choice between using `IndexWith`
        // and `IndirectIndexWith`.
        elem_id = ConvertToValueExpression(context, elem_id);
      }
      context.node_stack().Push(parse_node, elem_id);
      return true;
    }
    case SemIR::TupleType::Kind: {
      SemIR::TypeId element_type_id = SemIR::TypeId::Error;
      if (auto index_literal = index_node.TryAs<SemIR::IntegerLiteral>()) {
        auto type_block = context.semantics_ir().GetTypeBlock(
            operand_type_node.As<SemIR::TupleType>().elements_id);
        if (const auto* index_val = ValidateIntegerLiteralBound(
                context, parse_node, operand_node, *index_literal,
                type_block.size())) {
          element_type_id = type_block[index_val->getZExtValue()];
        } else {
          index_node_id = SemIR::NodeId::BuiltinError;
        }
      } else if (index_node.type_id() != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TupleIndexIntegerLiteral, Error,
                          "Tuples indices must be integer literals.");
        context.emitter().Emit(parse_node, TupleIndexIntegerLiteral);
        index_node_id = SemIR::NodeId::BuiltinError;
      }
      context.AddNodeAndPush(parse_node,
                             SemIR::TupleIndex(parse_node, element_type_id,
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

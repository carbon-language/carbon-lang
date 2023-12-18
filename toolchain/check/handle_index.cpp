// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APSInt.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleIndexExprStart(Context& /*context*/, Parse::NodeId /*parse_node*/)
    -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

// Validates that the index (required to be an IntLiteral) is valid within
// the array or tuple size. Returns the index on success, or nullptr on failure.
static auto ValidateIntLiteralBound(Context& context, Parse::NodeId parse_node,
                                    SemIR::Inst operand_inst,
                                    SemIR::IntLiteral index_inst, int size)
    -> const llvm::APInt* {
  const auto& index_val = context.ints().Get(index_inst.int_id);
  if (index_val.uge(size)) {
    CARBON_DIAGNOSTIC(IndexOutOfBounds, Error,
                      "Index `{0}` is past the end of type `{1}`.",
                      llvm::APSInt, std::string);
    context.emitter().Emit(
        parse_node, IndexOutOfBounds,
        llvm::APSInt(index_val, /*isUnsigned=*/true),
        context.sem_ir().StringifyType(operand_inst.type_id()));
    return nullptr;
  }
  return &index_val;
}

auto HandleIndexExpr(Context& context, Parse::NodeId parse_node) -> bool {
  auto index_inst_id = context.node_stack().PopExpr();
  auto index_inst = context.insts().Get(index_inst_id);
  auto operand_inst_id = context.node_stack().PopExpr();
  operand_inst_id = ConvertToValueOrRefExpr(context, operand_inst_id);
  auto operand_inst = context.insts().Get(operand_inst_id);
  auto operand_type_id = operand_inst.type_id();
  auto operand_type_inst = context.types().GetAsInst(operand_type_id);

  switch (operand_type_inst.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = operand_type_inst.As<SemIR::ArrayType>();
      // We can check whether integers are in-bounds, although it doesn't affect
      // the IR for an array.
      if (auto index_literal = index_inst.TryAs<SemIR::IntLiteral>();
          index_literal &&
          !ValidateIntLiteralBound(
              context, parse_node, operand_inst, *index_literal,
              context.sem_ir().GetArrayBoundValue(array_type.bound_id))) {
        index_inst_id = SemIR::InstId::BuiltinError;
      }
      auto cast_index_id = ConvertToValueOfType(
          context, index_inst.parse_node(), index_inst_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntType));
      auto array_cat =
          SemIR::GetExprCategory(context.sem_ir(), operand_inst_id);
      if (array_cat == SemIR::ExprCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_inst_id = context.AddInst(
            SemIR::ValueAsRef{parse_node, operand_type_id, operand_inst_id});
      }
      auto elem_id = context.AddInst(
          SemIR::ArrayIndex{parse_node, array_type.element_type_id,
                            operand_inst_id, cast_index_id});
      if (array_cat != SemIR::ExprCategory::DurableRef) {
        // Indexing a durable reference gives a durable reference expression.
        // Indexing anything else gives a value expression.
        // TODO: This should be replaced by a choice between using `IndexWith`
        // and `IndirectIndexWith`.
        elem_id = ConvertToValueExpr(context, elem_id);
      }
      context.node_stack().Push(parse_node, elem_id);
      return true;
    }
    case SemIR::TupleType::Kind: {
      SemIR::TypeId element_type_id = SemIR::TypeId::Error;
      if (auto index_literal = index_inst.TryAs<SemIR::IntLiteral>()) {
        auto type_block = context.type_blocks().Get(
            operand_type_inst.As<SemIR::TupleType>().elements_id);
        if (const auto* index_val =
                ValidateIntLiteralBound(context, parse_node, operand_inst,
                                        *index_literal, type_block.size())) {
          element_type_id = type_block[index_val->getZExtValue()];
        } else {
          index_inst_id = SemIR::InstId::BuiltinError;
        }
      } else if (index_inst.type_id() != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TupleIndexIntLiteral, Error,
                          "Tuples indices must be integer literals.");
        context.emitter().Emit(parse_node, TupleIndexIntLiteral);
        index_inst_id = SemIR::InstId::BuiltinError;
      }
      context.AddInstAndPush(parse_node,
                             SemIR::TupleIndex{parse_node, element_type_id,
                                               operand_inst_id, index_inst_id});
      return true;
    }
    default: {
      if (operand_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TypeNotIndexable, Error,
                          "Type `{0}` does not support indexing.", std::string);
        context.emitter().Emit(parse_node, TypeNotIndexable,
                               context.sem_ir().StringifyType(operand_type_id));
      }
      context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
      return true;
    }
  }
}

}  // namespace Carbon::Check

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/APSInt.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleIndexExprStart(Context& /*context*/,
                          Parse::IndexExprStartId /*parse_node*/) -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

// Validates that the index (required to be an IntLiteral) is valid within the
// tuple size. Returns the index on success, or nullptr on failure.
static auto ValidateTupleIndex(Context& context, Parse::NodeId parse_node,
                               SemIR::Inst operand_inst,
                               SemIR::IntLiteral index_inst, int size)
    -> const llvm::APInt* {
  const auto& index_val = context.ints().Get(index_inst.int_id);
  if (index_val.uge(size)) {
    CARBON_DIAGNOSTIC(
        TupleIndexOutOfBounds, Error,
        "Tuple element index `{0}` is past the end of type `{1}`.",
        llvm::APSInt, std::string);
    context.emitter().Emit(
        parse_node, TupleIndexOutOfBounds,
        llvm::APSInt(index_val, /*isUnsigned=*/true),
        context.sem_ir().StringifyType(operand_inst.type_id()));
    return nullptr;
  }
  return &index_val;
}

auto HandleIndexExpr(Context& context, Parse::IndexExprId parse_node) -> bool {
  auto index_inst_id = context.node_stack().PopExpr();
  auto operand_inst_id = context.node_stack().PopExpr();
  operand_inst_id = ConvertToValueOrRefExpr(context, operand_inst_id);
  auto operand_inst = context.insts().Get(operand_inst_id);
  auto operand_type_id = operand_inst.type_id();
  auto operand_type_inst = context.types().GetAsInst(operand_type_id);

  switch (operand_type_inst.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = operand_type_inst.As<SemIR::ArrayType>();
      auto index_parse_node = context.insts().GetParseNode(index_inst_id);
      auto cast_index_id = ConvertToValueOfType(
          context, index_parse_node, index_inst_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntType));
      auto array_cat =
          SemIR::GetExprCategory(context.sem_ir(), operand_inst_id);
      if (array_cat == SemIR::ExprCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_inst_id = context.AddInst(
            {parse_node, SemIR::ValueAsRef{operand_type_id, operand_inst_id}});
      }
      // Constant evaluation will perform a bounds check on this array indexing
      // if the index is constant.
      auto elem_id = context.AddInst(
          {parse_node, SemIR::ArrayIndex{array_type.element_type_id,
                                         operand_inst_id, cast_index_id}});
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
      auto index_parse_node = context.insts().GetParseNode(index_inst_id);
      index_inst_id = ConvertToValueOfType(
          context, index_parse_node, index_inst_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntType));
      auto index_const_id = context.constant_values().Get(index_inst_id);
      if (index_const_id == SemIR::ConstantId::Error) {
        index_inst_id = SemIR::InstId::BuiltinError;
      } else if (!index_const_id.is_template()) {
        // TODO: Decide what to do if the index is a symbolic constant.
        CARBON_DIAGNOSTIC(TupleIndexNotConstant, Error,
                          "Tuple index must be a constant.");
        context.emitter().Emit(parse_node, TupleIndexNotConstant);
        index_inst_id = SemIR::InstId::BuiltinError;
      } else {
        auto index_literal =
            context.insts().GetAs<SemIR::IntLiteral>(index_const_id.inst_id());
        auto type_block = context.type_blocks().Get(
            operand_type_inst.As<SemIR::TupleType>().elements_id);
        if (const auto* index_val =
                ValidateTupleIndex(context, parse_node, operand_inst,
                                   index_literal, type_block.size())) {
          element_type_id = type_block[index_val->getZExtValue()];
        } else {
          index_inst_id = SemIR::InstId::BuiltinError;
        }
      }
      context.AddInstAndPush(
          {parse_node,
           SemIR::TupleIndex{element_type_id, operand_inst_id, index_inst_id}});
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

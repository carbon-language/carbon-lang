// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleIndexExprStart(Context& /*context*/,
                          Parse::IndexExprStartId /*node_id*/) -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

// Validates that the index (required to be an IntLiteral) is valid within the
// tuple size. Returns the index on success, or nullptr on failure.
static auto ValidateTupleIndex(Context& context, Parse::NodeId node_id,
                               SemIR::Inst operand_inst,
                               SemIR::IntLiteral index_inst, int size)
    -> const llvm::APInt* {
  const auto& index_val = context.ints().Get(index_inst.int_id);
  if (index_val.uge(size)) {
    CARBON_DIAGNOSTIC(
        TupleIndexOutOfBounds, Error,
        "Tuple element index `{0}` is past the end of type `{1}`.", TypedInt,
        SemIR::TypeId);
    context.emitter().Emit(node_id, TupleIndexOutOfBounds,
                           TypedInt{index_inst.type_id, index_val},
                           operand_inst.type_id());
    return nullptr;
  }
  return &index_val;
}

auto HandleIndexExpr(Context& context, Parse::IndexExprId node_id) -> bool {
  auto index_inst_id = context.node_stack().PopExpr();
  auto operand_inst_id = context.node_stack().PopExpr();
  operand_inst_id = ConvertToValueOrRefExpr(context, operand_inst_id);
  auto operand_inst = context.insts().Get(operand_inst_id);
  auto operand_type_id = operand_inst.type_id();
  CARBON_KIND_SWITCH(context.types().GetAsInst(operand_type_id)) {
    case CARBON_KIND(SemIR::ArrayType array_type): {
      auto index_node_id = context.insts().GetLocId(index_inst_id);
      auto cast_index_id = ConvertToValueOfType(
          context, index_node_id, index_inst_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntType));
      auto array_cat =
          SemIR::GetExprCategory(context.sem_ir(), operand_inst_id);
      if (array_cat == SemIR::ExprCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_inst_id = context.AddInst(
            {node_id, SemIR::ValueAsRef{operand_type_id, operand_inst_id}});
      }
      // Constant evaluation will perform a bounds check on this array indexing
      // if the index is constant.
      auto elem_id = context.AddInst(
          {node_id, SemIR::ArrayIndex{array_type.element_type_id,
                                      operand_inst_id, cast_index_id}});
      if (array_cat != SemIR::ExprCategory::DurableRef) {
        // Indexing a durable reference gives a durable reference expression.
        // Indexing anything else gives a value expression.
        // TODO: This should be replaced by a choice between using `IndexWith`
        // and `IndirectIndexWith`.
        elem_id = ConvertToValueExpr(context, elem_id);
      }
      context.node_stack().Push(node_id, elem_id);
      return true;
    }
    case CARBON_KIND(SemIR::TupleType tuple_type): {
      SemIR::TypeId element_type_id = SemIR::TypeId::Error;
      auto index_node_id = context.insts().GetLocId(index_inst_id);
      index_inst_id = ConvertToValueOfType(
          context, index_node_id, index_inst_id,
          context.GetBuiltinType(SemIR::BuiltinKind::IntType));
      auto index_const_id = context.constant_values().Get(index_inst_id);
      if (index_const_id == SemIR::ConstantId::Error) {
        index_inst_id = SemIR::InstId::BuiltinError;
      } else if (!index_const_id.is_template()) {
        // TODO: Decide what to do if the index is a symbolic constant.
        CARBON_DIAGNOSTIC(TupleIndexNotConstant, Error,
                          "Tuple index must be a constant.");
        context.emitter().Emit(node_id, TupleIndexNotConstant);
        index_inst_id = SemIR::InstId::BuiltinError;
      } else {
        auto index_literal =
            context.insts().GetAs<SemIR::IntLiteral>(index_const_id.inst_id());
        auto type_block = context.type_blocks().Get(tuple_type.elements_id);
        if (const auto* index_val =
                ValidateTupleIndex(context, node_id, operand_inst,
                                   index_literal, type_block.size())) {
          element_type_id = type_block[index_val->getZExtValue()];
        } else {
          index_inst_id = SemIR::InstId::BuiltinError;
        }
      }
      context.AddInstAndPush(
          {node_id,
           SemIR::TupleIndex{element_type_id, operand_inst_id, index_inst_id}});
      return true;
    }
    default: {
      if (operand_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TypeNotIndexable, Error,
                          "Type `{0}` does not support indexing.",
                          SemIR::TypeId);
        context.emitter().Emit(node_id, TypeNotIndexable, operand_type_id);
      }
      context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
      return true;
    }
  }
}

}  // namespace Carbon::Check

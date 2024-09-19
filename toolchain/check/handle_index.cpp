// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/, Parse::IndexExprStartId /*node_id*/)
    -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

auto HandleParseNode(Context& context, Parse::IndexExprId node_id) -> bool {
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
          context.GetBuiltinType(SemIR::BuiltinInstKind::IntType));
      auto array_cat =
          SemIR::GetExprCategory(context.sem_ir(), operand_inst_id);
      if (array_cat == SemIR::ExprCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_inst_id = context.AddInst<SemIR::ValueAsRef>(
            node_id, {.type_id = operand_type_id, .value_id = operand_inst_id});
      }
      // Constant evaluation will perform a bounds check on this array indexing
      // if the index is constant.
      auto elem_id = context.AddInst<SemIR::ArrayIndex>(
          node_id, {.type_id = array_type.element_type_id,
                    .array_id = operand_inst_id,
                    .index_id = cast_index_id});
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
    default: {
      if (operand_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(TypeNotIndexable, Error,
                          "type `{0}` does not support indexing",
                          SemIR::TypeId);
        context.emitter().Emit(node_id, TypeNotIndexable, operand_type_id);
      }
      context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
      return true;
    }
  }
}

}  // namespace Carbon::Check

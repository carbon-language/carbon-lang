// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/, Parse::IndexExprStartId /*node_id*/)
    -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

// Performs an index with base expression `operand_inst_id` and
// `operand_type_id` for types that are not an array. This checks if
// the base expression implements the `IndexWith` interface; if so, uses the
// `At` associative method, otherwise prints a diagnostic.
static auto PerformIndexWith(Context& context, Parse::NodeId node_id,
                             SemIR::InstId operand_inst_id,
                             SemIR::InstId index_inst_id) -> SemIR::InstId {
  SemIR::InstId args[] = {
      context.types().GetInstId(context.insts().Get(index_inst_id).type_id())};
  Operator op{.interface_name = "IndexWith",
              .interface_args_ref = args,
              .op_name = "At"};
  return BuildBinaryOperator(context, node_id, op, operand_inst_id,
                             index_inst_id);
}

auto HandleParseNode(Context& context, Parse::IndexExprId node_id) -> bool {
  auto index_inst_id = context.node_stack().PopExpr();
  auto operand_inst_id = context.node_stack().PopExpr();
  operand_inst_id = ConvertToValueOrRefExpr(context, operand_inst_id);
  auto operand_inst = context.insts().Get(operand_inst_id);
  auto operand_type_id = operand_inst.type_id();

  CARBON_KIND_SWITCH(context.types().GetAsInst(operand_type_id)) {
    case CARBON_KIND(SemIR::ArrayType array_type): {
      auto cast_index_id = ConvertToValueOfType(
          context, SemIR::LocId(index_inst_id), index_inst_id,
          // TODO: Replace this with impl lookup rather than hardcoding `i32`.
          MakeIntType(context, node_id, SemIR::IntKind::Signed,
                      context.ints().Add(32)));
      auto array_cat =
          SemIR::GetExprCategory(context.sem_ir(), operand_inst_id);
      if (array_cat == SemIR::ExprCategory::Value) {
        // If the operand is an array value, convert it to an ephemeral
        // reference to an array so we can perform a primitive indexing into it.
        operand_inst_id = AddInst<SemIR::ValueAsRef>(
            context, node_id,
            {.type_id = operand_type_id, .value_id = operand_inst_id});
      }
      // Constant evaluation will perform a bounds check on this array indexing
      // if the index is constant.
      auto elem_id = AddInst<SemIR::ArrayIndex>(
          context, node_id,
          {.type_id = context.types().GetTypeIdForTypeInstId(
               array_type.element_type_inst_id),
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
      auto elem_id =
          PerformIndexWith(context, node_id, operand_inst_id, index_inst_id);
      context.node_stack().Push(node_id, elem_id);
      return true;
    }
  }
}

}  // namespace Carbon::Check

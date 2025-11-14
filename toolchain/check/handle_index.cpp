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

// Performs bounds checking for string indexing when the index is a constant.
static auto CheckStringIndexBounds(Context& context,
                                   SemIR::InstId operand_inst_id,
                                   SemIR::InstId index_inst_id,
                                   const llvm::APInt& index_int) -> void {
  if (index_int.isNegative()) {
    CARBON_DIAGNOSTIC(ArrayIndexNegative, Error, "index `{0}` is negative.",
                      TypedInt);
    context.emitter().Emit(
        SemIR::LocId(index_inst_id), ArrayIndexNegative,
        {.type = context.insts().Get(index_inst_id).type_id(),
         .value = index_int});
    return;
  }

  auto operand_const_id = context.constant_values().Get(operand_inst_id);
  if (!operand_const_id.is_constant()) {
    return;
  }

  auto operand_const_inst_id =
      context.constant_values().GetInstId(operand_const_id);
  auto str_struct =
      context.insts().TryGetAs<SemIR::StructValue>(operand_const_inst_id);
  if (!str_struct) {
    return;
  }

  auto elements = context.inst_blocks().Get(str_struct->elements_id);
  CARBON_CHECK(elements.size() == 2, "String struct should have 2 fields.");

  auto ptr_const_id = context.constant_values().Get(elements[0]);
  auto ptr_inst_id = context.constant_values().GetInstId(ptr_const_id);
  auto string_literal =
      context.insts().TryGetAs<SemIR::StringLiteral>(ptr_inst_id);
  if (!string_literal) {
    return;
  }

  auto string_value = context.sem_ir().string_literal_values().Get(
      string_literal->string_literal_id);
  if (index_int.getActiveBits() > 64 ||
      index_int.getZExtValue() >= string_value.size()) {
    CARBON_DIAGNOSTIC(StringAtIndexOutOfBounds, Error,
                      "string index `{0}` is past the end of the string.",
                      TypedInt);
    context.emitter().Emit(
        SemIR::LocId(index_inst_id), StringAtIndexOutOfBounds,
        {.type = context.insts().Get(index_inst_id).type_id(),
         .value = index_int});
  }
}

// Checks if the given ClassType is the String class.
static auto IsStringType(Context& context, SemIR::ClassType class_type)
    -> bool {
  auto& class_info = context.classes().Get(class_type.class_id);
  auto identifier_id = class_info.name_id.AsIdentifierId();
  return identifier_id.has_value() &&
         context.identifiers().Get(identifier_id) == "String";
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

    case CARBON_KIND(SemIR::ClassType class_type): {
      if (IsStringType(context, class_type)) {
        auto index_const_id = context.constant_values().Get(index_inst_id);
        if (index_const_id.is_constant()) {
          auto index_const_inst_id =
              context.constant_values().GetInstId(index_const_id);
          if (auto index_val = context.insts().TryGetAs<SemIR::IntValue>(
                  index_const_inst_id)) {
            const auto& index_int = context.ints().Get(index_val->int_id);
            CheckStringIndexBounds(context, operand_inst_id, index_inst_id,
                                   index_int);
          }
        }
      }

      auto elem_id =
          PerformIndexWith(context, node_id, operand_inst_id, index_inst_id);
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

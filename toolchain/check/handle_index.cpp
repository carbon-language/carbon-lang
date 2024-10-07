// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/operator.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/builtin_inst_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/, Parse::IndexExprStartId /*node_id*/)
    -> bool {
  // Leave the expression on the stack for IndexExpr.
  return true;
}

// Returns the argument values of the `IndexWith` interface, this corresponds to
// the `SubscriptType` and the `ElementType`, if the class does not implement
// the said interface this returns an empty array reference.
static auto GetIndexWithArgs(Context& context, Parse::NodeId node_id,
                             SemIR::TypeId self_id)
    -> llvm::ArrayRef<SemIR::InstId> {
  auto index_with_interface =
      context.types().GetAs<SemIR::GenericInterfaceType>(
          context.insts()
              .GetAs<SemIR::StructValue>(
                  context.LookupNameInCore(node_id, "IndexWith"))
              .type_id);

  for (const auto& impl : context.impls().array_ref()) {
    if (impl.self_id != self_id) {
      continue;
    }
    auto interface_type =
        context.types().TryGetAs<SemIR::InterfaceType>(impl.constraint_id);
    if (!interface_type) {
      continue;
    }

    if (index_with_interface.interface_id != interface_type->interface_id) {
      continue;
    }

    return context.inst_blocks().GetOrEmpty(
        context.specifics().Get(interface_type->specific_id).args_id);
  }

  return {};
}

static auto PerformIndex(Context& context, Parse::NodeId node_id,
                         SemIR::InstId operand_inst_id,
                         SemIR::TypeId operand_type_id,
                         SemIR::InstId index_inst_id) -> SemIR::InstId {
  auto args = GetIndexWithArgs(context, node_id, operand_type_id);

  CARBON_DIAGNOSTIC(TypeNotIndexable, Error,
                    "type `{0}` does not support indexing", SemIR::TypeId);

  // If the type does not implement the `IndexWith` interface, then return
  // an error.
  if (args.empty()) {
    context.emitter().Emit(node_id, TypeNotIndexable, operand_type_id);
    return SemIR::InstId::BuiltinError;
  }

  CARBON_CHECK(args.size() == 2,
               "IndexWith should have two generic constraints");

  auto op = Operator{
      .interface_name = "IndexWith",
      .interface_args_ref = args,
      .op_name = "At",
  };

  // The first argument of the `IndexWith` interface corresponds to the
  // `SubscriptType`, so first cast `index_inst_id` to that type.
  auto subscript_type_id = context.GetTypeIdForTypeInst(args[0]);
  auto cast_index =
      ConvertToValueOfType(context, node_id, index_inst_id, subscript_type_id);

  auto result =
      BuildBinaryOperator(context, node_id, op, operand_inst_id, cast_index);

  return result;
}

auto HandleParseNode(Context& context, Parse::IndexExprId node_id) -> bool {
  auto index_inst_id = context.node_stack().PopExpr();
  auto operand_inst_id = context.node_stack().PopExpr();
  operand_inst_id = ConvertToValueOrRefExpr(context, operand_inst_id);
  auto operand_inst = context.insts().Get(operand_inst_id);
  auto operand_type_id = operand_inst.type_id();

  CARBON_KIND_SWITCH(context.types().GetAsInst(operand_type_id)) {
    case CARBON_KIND(SemIR::ArrayType array_type): {
      auto index_loc_id = context.insts().GetLocId(index_inst_id);
      auto cast_index_id = ConvertToValueOfType(
          context, index_loc_id, index_inst_id,
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
      auto elem_id = SemIR::InstId::BuiltinError;
      if (operand_type_id != SemIR::TypeId::Error) {
        elem_id = PerformIndex(context, node_id, operand_inst_id,
                               operand_type_id, index_inst_id);
      }
      context.node_stack().Push(node_id, elem_id);
      return true;
    }
  }
}

}  // namespace Carbon::Check

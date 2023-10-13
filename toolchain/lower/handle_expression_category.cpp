// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Lower {

auto HandleBindValue(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::BindValue node) -> void {
  switch (auto rep = SemIR::GetValueRepresentation(context.semantics_ir(),
                                                   node.type_id);
          rep.kind) {
    case SemIR::ValueRepresentation::Unknown:
      CARBON_FATAL()
          << "Value binding for type with incomplete value representation";
    case SemIR::ValueRepresentation::None:
      // Nothing should use this value, but StubReference needs a value to
      // propagate.
      // TODO: Remove this now the StubReferences are gone.
      context.SetLocal(node_id,
                       llvm::PoisonValue::get(context.GetType(node.type_id)));
      break;
    case SemIR::ValueRepresentation::Copy:
      context.SetLocal(node_id, context.builder().CreateLoad(
                                    context.GetType(node.type_id),
                                    context.GetLocal(node.value_id)));
      break;
    case SemIR::ValueRepresentation::Pointer:
      context.SetLocal(node_id, context.GetLocal(node.value_id));
      break;
    case SemIR::ValueRepresentation::Custom:
      CARBON_FATAL() << "TODO: Add support for BindValue with custom value rep";
  }
}

auto HandleTemporary(FunctionContext& context, SemIR::NodeId node_id,
                     SemIR::Temporary node) -> void {
  context.FinishInitialization(node.type_id, node.storage_id, node.init_id);
  context.SetLocal(node_id, context.GetLocal(node.storage_id));
}

auto HandleTemporaryStorage(FunctionContext& context, SemIR::NodeId node_id,
                            SemIR::TemporaryStorage node) -> void {
  context.SetLocal(node_id,
                   context.builder().CreateAlloca(context.GetType(node.type_id),
                                                  nullptr, "temp"));
}

auto HandleValueAsReference(FunctionContext& context, SemIR::NodeId node_id,
                            SemIR::ValueAsReference node) -> void {
  CARBON_CHECK(
      SemIR::GetExpressionCategory(context.semantics_ir(), node.value_id) ==
      SemIR::ExpressionCategory::Value);
  CARBON_CHECK(
      SemIR::GetValueRepresentation(context.semantics_ir(), node.type_id)
          .kind == SemIR::ValueRepresentation::Pointer);
  context.SetLocal(node_id, context.GetLocal(node.value_id));
}

}  // namespace Carbon::Lower
